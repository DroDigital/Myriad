//! GPU integration tests. They need a GPU adapter; on machines without one
//! (and without a software driver such as Mesa's lavapipe) they are skipped,
//! unless `MYRIAD_REQUIRE_GPU=1` is set (CI sets it).

use glam::Vec2;
use myriad::game::Controls;
use myriad::gpu::Gpu;
use myriad::headless::{self, CAPTURE_FORMAT, HeadlessOptions};
use myriad::session::{Session, SessionConfig};
use myriad::sim::stats::SwarmStats;
use myriad::sim::{Agent, INVALID_CELL, SimFrame, Swarm, SwarmConfig, WorldBounds, spawn_disk};
use myriad::util::SplitMix64;

fn gpu() -> Option<Gpu> {
    match Gpu::headless() {
        Ok(gpu) => Some(gpu),
        Err(e) if std::env::var("MYRIAD_REQUIRE_GPU").is_ok_and(|v| v == "1") => {
            panic!("MYRIAD_REQUIRE_GPU=1 but no GPU adapter: {e:#}")
        }
        Err(e) => {
            eprintln!("skipping GPU test: {e:#}");
            None
        }
    }
}

fn bounds() -> WorldBounds {
    WorldBounds {
        min: Vec2::new(-60.0, -40.0),
        max: Vec2::new(60.0, 40.0),
    }
}

fn swarm_with(gpu: &Gpu, agents: &[Agent], capacity: u32) -> Swarm {
    let mut swarm = Swarm::new(
        &gpu.device,
        SwarmConfig {
            capacity,
            bounds: bounds(),
            cell_size: 1.0,
        },
    );
    swarm.spawn(&gpu.queue, agents);
    swarm
}

fn frame(focal: Vec2, step: u32) -> SimFrame {
    SimFrame {
        focal,
        gather: 0.0,
        dt: 1.0 / 60.0,
        time: step as f32 / 60.0,
    }
}

#[test]
fn spatial_hash_buckets_every_agent_exactly_once() {
    let Some(gpu) = gpu() else { return };
    let mut rng = SplitMix64::new(3);
    // Scatter across the world, including exactly on and beyond the edges.
    let mut agents: Vec<Agent> = (0..5_000)
        .map(|_| {
            let p = Vec2::new(rng.next_f32() * 130.0 - 65.0, rng.next_f32() * 90.0 - 45.0);
            Agent::new(p, rng.next_f32())
        })
        .collect();
    agents[7].flags = 0; // one dead agent must be left out
    let mut swarm = swarm_with(&gpu, &agents, 8_192);
    swarm.step(&gpu, &frame(Vec2::ZERO, 0));
    let grid = swarm.read_grid(&gpu).unwrap();

    let total: u32 = grid.counts.iter().sum();
    assert_eq!(total, 4_999);
    // Starts are the exclusive prefix sum of counts.
    let mut running = 0;
    for (c, (&start, &count)) in grid.starts.iter().zip(&grid.counts).enumerate() {
        assert_eq!(start, running, "cell {c}");
        running += count;
    }
    // Each living agent appears once, inside its own cell's slice.
    let mut seen = vec![false; agents.len()];
    for (c, (&start, &count)) in grid.starts.iter().zip(&grid.counts).enumerate() {
        for &i in &grid.sorted[start as usize..(start + count) as usize] {
            let i = i as usize;
            assert!(!seen[i], "agent {i} listed twice");
            seen[i] = true;
            assert_eq!(grid.agent_cell[i], c as u32);
            assert_eq!(swarm.cell_of(agents[i].pos()), c as u32, "agent {i}");
        }
    }
    assert!(!seen[7]);
    assert_eq!(grid.agent_cell[7], INVALID_CELL);
    assert_eq!(seen.iter().filter(|&&s| s).count(), 4_999);
}

#[test]
fn swarm_flows_to_the_focal_point_and_stays_healthy() {
    let Some(gpu) = gpu() else { return };
    let mut rng = SplitMix64::new(5);
    let start = Vec2::new(-30.0, 0.0);
    let focal = Vec2::new(30.0, 10.0);
    let agents = spawn_disk(start, 3_000, 0.5, &mut rng);
    let mut swarm = swarm_with(&gpu, &agents, 4_096);

    let before = SwarmStats::compute(&swarm.read_agents(&gpu).unwrap(), focal, bounds());
    // 60+ units at a top speed of 9: allow 15 s to arrive and settle.
    for step in 0..900 {
        swarm.step(&gpu, &frame(focal, step));
    }
    let after = SwarmStats::compute(&swarm.read_agents(&gpu).unwrap(), focal, bounds());
    eprintln!("before: {before}\nafter:  {after}");

    assert_eq!(after.alive, 3_000);
    assert_eq!(after.non_finite, 0);
    assert_eq!(after.out_of_bounds, 0);
    // It arrived: the centroid is near the focal point...
    assert!((after.centroid - focal).length() < 3.0, "{after}");
    // ...pooled as a disk around it rather than a crush or a scatter. A disk of
    // 3000 at rest density has radius ~15; the mean distance is ~2/3 of that.
    assert!((4.0..16.0).contains(&after.mean_dist_to_focal), "{after}");
    assert!(after.mean_nearest > 0.1, "swarm collapsed: {after}");
}

#[test]
fn gather_tightens_and_spread_loosens() {
    let Some(gpu) = gpu() else { return };
    let focal = Vec2::ZERO;
    let settle = |gather: f32| {
        let mut rng = SplitMix64::new(9);
        let agents = spawn_disk(focal, 2_000, 0.5, &mut rng);
        let mut swarm = swarm_with(&gpu, &agents, 2_048);
        for step in 0..480 {
            let mut f = frame(focal, step);
            f.gather = gather;
            swarm.step(&gpu, &f);
        }
        SwarmStats::compute(&swarm.read_agents(&gpu).unwrap(), focal, bounds())
    };
    let neutral = settle(0.0);
    let gathered = settle(1.0);
    let spread = settle(-1.0);
    eprintln!("spread:   {spread}\nneutral:  {neutral}\ngathered: {gathered}");
    assert!(gathered.mean_dist_to_focal < neutral.mean_dist_to_focal);
    assert!(spread.mean_dist_to_focal > neutral.mean_dist_to_focal);
}

#[test]
fn feedback_reports_the_centroid_without_blocking() {
    let Some(gpu) = gpu() else { return };
    let mut rng = SplitMix64::new(11);
    let agents = spawn_disk(Vec2::new(10.0, -5.0), 1_000, 0.5, &mut rng);
    let mut swarm = swarm_with(&gpu, &agents, 1_024);
    let mut summary = None;
    for step in 0..60 {
        swarm.step(&gpu, &frame(Vec2::new(10.0, -5.0), step));
        summary = swarm.summary(&gpu).or(summary);
    }
    gpu.wait_idle();
    let summary = swarm
        .summary(&gpu)
        .or(summary)
        .expect("no feedback arrived");
    let exact = SwarmStats::compute(&swarm.read_agents(&gpu).unwrap(), Vec2::ZERO, bounds());
    assert_eq!(summary.alive, 1_000);
    // A frame or two of latency: close to, not necessarily equal to, now.
    assert!(
        (summary.centroid - exact.centroid).length() < 0.5,
        "{summary:?} vs {exact}"
    );
    assert!(summary.step > 50);
}

#[test]
fn capacity_is_respected() {
    let Some(gpu) = gpu() else { return };
    let mut rng = SplitMix64::new(1);
    let agents = spawn_disk(Vec2::ZERO, 100, 0.5, &mut rng);
    let mut swarm = swarm_with(&gpu, &agents, 150);
    assert_eq!(swarm.spawn(&gpu.queue, &agents), 50);
    assert_eq!(swarm.count(), 150);
    assert_eq!(swarm.spawn(&gpu.queue, &agents), 0);
}

#[test]
fn session_hatches_and_renders_a_frame() {
    let Some(gpu) = gpu() else { return };
    let cfg = SessionConfig {
        initial_agents: 500,
        capacity: 2_000,
        ..Default::default()
    };
    let mut session = Session::new(&gpu, CAPTURE_FORMAT, 16.0 / 9.0, &cfg);
    assert_eq!(session.hatch(&gpu, 10_000), 1_500);
    for _ in 0..30 {
        session.step(&gpu, &Controls::default());
    }
    let image = headless::capture(&gpu, &session, 320, 180).unwrap();
    // The swarm sits at the centre of the view: dark critters on a light floor.
    let center = image.get_pixel(160, 90);
    let corner = image.get_pixel(2, 2);
    let luma = |p: &image::Rgba<u8>| p.0[0] as u32 + p.0[1] as u32 + p.0[2] as u32;
    assert!(
        luma(center) < luma(corner),
        "center {center:?} corner {corner:?}"
    );
}

#[test]
fn headless_run_reports_sane_numbers() {
    if gpu().is_none() {
        return;
    }
    let report = headless::run(&HeadlessOptions {
        agents: 4_000,
        frames: 120,
        ..Default::default()
    })
    .unwrap();
    assert_eq!(report.agents, 4_000);
    assert_eq!(report.stats.non_finite, 0);
    assert!(report.ms_per_step > 0.0);
}
