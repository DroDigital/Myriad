//! GPU integration tests. They need a GPU adapter; on machines without one
//! (and without a software driver such as Mesa's lavapipe) they are skipped,
//! unless `MYRIAD_REQUIRE_GPU=1` is set (CI sets it).

use glam::Vec2;
use myriad::game::Controls;
use myriad::gpu::Gpu;
use myriad::headless::{self, CAPTURE_FORMAT, HeadlessOptions};
use myriad::run::RunConfig;
use myriad::session::{Session, SessionConfig};
use myriad::sim::material::{FLAG_BLOCKED, FULL, KINDS, Kind, MaterialGrid, MaterialLayout};
use myriad::sim::stats::SwarmStats;
use myriad::sim::{
    Agent, HazardZone, INVALID_CELL, SimFrame, Swarm, SwarmConfig, WorldBounds, counter, spawn_disk,
};
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

fn swarm_on(
    gpu: &Gpu,
    layout: &MaterialLayout,
    agents: &[Agent],
    capacity: u32,
) -> (Swarm, MaterialGrid) {
    let material = MaterialGrid::new(gpu, layout);
    let mut swarm = Swarm::new(
        gpu,
        SwarmConfig {
            capacity,
            bounds: layout.bounds,
            cell_size: 1.0,
        },
        &material,
    );
    swarm.spawn(&gpu.queue, agents);
    (swarm, material)
}

fn swarm_with(gpu: &Gpu, agents: &[Agent], capacity: u32) -> Swarm {
    swarm_on(gpu, &MaterialLayout::empty(bounds()), agents, capacity).0
}

fn frame(focal: Vec2, step: u32) -> SimFrame {
    SimFrame::steer(focal, 1.0 / 60.0, step as f32 / 60.0)
}

fn alive(swarm: &Swarm, gpu: &Gpu) -> Vec<Agent> {
    swarm
        .read_agents(gpu)
        .unwrap()
        .into_iter()
        .filter(Agent::is_alive)
        .collect()
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
    let focal = Vec2::new(30.0, 10.0);
    let agents = spawn_disk(Vec2::new(-30.0, 0.0), 3_000, 0.5, &mut rng);
    let mut swarm = swarm_with(&gpu, &agents, 4_096);
    // 60+ units at a top speed of 9: allow 15 s to arrive and settle.
    for step in 0..900 {
        swarm.step(&gpu, &frame(focal, step));
    }
    let after = SwarmStats::compute(&swarm.read_agents(&gpu).unwrap(), focal, bounds());
    eprintln!("after: {after}");
    assert_eq!(after.alive, 3_000);
    assert_eq!(after.non_finite, 0);
    assert_eq!(after.out_of_bounds, 0);
    // It arrived and pooled as a disk around the focal point: neither a crush
    // nor a scatter. A disk of 3000 at rest density has radius ~20.
    assert!((after.centroid - focal).length() < 3.0, "{after}");
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
fn feedback_reports_the_swarm_without_blocking() {
    let Some(gpu) = gpu() else { return };
    let mut rng = SplitMix64::new(11);
    let center = Vec2::new(10.0, -5.0);
    let agents = spawn_disk(center, 1_000, 0.5, &mut rng);
    let mut swarm = swarm_with(&gpu, &agents, 1_024);
    let mut summary = None;
    for step in 0..60 {
        let mut f = frame(center, step);
        f.queen_pos = center;
        swarm.step(&gpu, &f);
        summary = swarm.summary(&gpu).or(summary);
    }
    gpu.wait_idle();
    let summary = swarm
        .summary(&gpu)
        .or(summary)
        .expect("no feedback arrived");
    let exact = SwarmStats::compute(&swarm.read_agents(&gpu).unwrap(), Vec2::ZERO, bounds());
    assert_eq!(
        (summary.alive, summary.body, summary.live_end),
        (1_000, 1_000, 1_000)
    );
    assert!(summary.near_queen > 20, "{summary:?}");
    // A frame or two of latency: close to, not necessarily equal to, now.
    assert!(
        (summary.centroid - exact.centroid).length() < 0.5,
        "{summary:?} vs {exact}"
    );
    assert!(summary.step > 50);
}

#[test]
fn eating_conserves_food_exactly() {
    let Some(gpu) = gpu() else { return };
    let mut layout = MaterialLayout::empty(bounds());
    // A jam spill underfoot and a bread block to push into.
    layout.fill_rect(Vec2::new(-6.0, -6.0), Vec2::new(6.0, 6.0), Kind::Jam, FULL);
    layout.fill_rect(
        Vec2::new(8.0, -3.0),
        Vec2::new(12.0, 3.0),
        Kind::Bread,
        FULL,
    );
    let initial: u64 = layout.total_food();
    let mut rng = SplitMix64::new(2);
    let agents = spawn_disk(Vec2::ZERO, 400, 0.5, &mut rng);
    let (mut swarm, material) = swarm_on(&gpu, &layout, &agents, 512);
    for step in 0..240 {
        // Steer into the bread so the solid gets gnawed too.
        swarm.step(&gpu, &frame(Vec2::new(10.0, 0.0), step));
    }
    let amounts = material.read_amounts(&gpu).unwrap();
    let sum_of = |kind: Option<Kind>| -> u64 {
        amounts
            .iter()
            .zip(&layout.kinds)
            .filter(|&(_, &k)| match kind {
                Some(kind) => k == kind as u32,
                None => k != Kind::Floor as u32,
            })
            .map(|(&a, _)| a as u64)
            .sum()
    };
    let eaten = swarm.read_counters(&gpu).unwrap()[counter::EATEN] as u64;
    assert!(eaten > 0, "nothing was eaten");
    assert_eq!(
        initial - sum_of(None),
        eaten,
        "food eaten must equal food removed"
    );
    // Some bread was gnawed away (solid food is eaten from the side).
    let bread_cells = layout
        .kinds
        .iter()
        .filter(|&&k| k == Kind::Bread as u32)
        .count() as u64;
    assert!(
        sum_of(Some(Kind::Bread)) < bread_cells * FULL as u64,
        "no bread was eaten"
    );
}

#[test]
fn critters_pressed_against_solid_food_keep_eating_through_it() {
    let Some(gpu) = gpu() else { return };
    let mut layout = MaterialLayout::empty(bounds());
    // A thick bread wall spanning the world's height: no way around it.
    layout.fill_rect(
        Vec2::new(4.0, -40.0),
        Vec2::new(16.0, 40.0),
        Kind::Bread,
        FULL,
    );
    let mut rng = SplitMix64::new(12);
    let agents = spawn_disk(Vec2::new(-4.0, 0.0), 600, 0.5, &mut rng);
    let (mut swarm, _material) = swarm_on(&gpu, &layout, &agents, 1_024);
    let focal = Vec2::new(30.0, 0.0);
    let eaten_by = |swarm: &Swarm| swarm.read_counters(&gpu).unwrap()[counter::EATEN];
    for step in 0..300 {
        swarm.step(&gpu, &frame(focal, step));
    }
    // Once the swarm is pressed flat against the wall it must keep eating,
    // not bounce off and stall.
    let before = eaten_by(&swarm);
    for step in 300..900 {
        swarm.step(&gpu, &frame(focal, step));
    }
    let gnawed_cells = (eaten_by(&swarm) - before) as f32 / FULL as f32;
    assert!(
        // Biting along the bounced-back velocity managed ~30 here; biting the
        // cell actually bumped into manages ~650.
        gnawed_cells > 200.0,
        "only {gnawed_cells:.1} cells eaten in 10 s against the wall"
    );
}

#[test]
fn critters_never_enter_walls() {
    let Some(gpu) = gpu() else { return };
    let mut layout = MaterialLayout::empty(bounds());
    // A wall between the swarm and the focal point.
    layout.fill_rect(Vec2::new(-1.0, -15.0), Vec2::new(1.0, 15.0), Kind::Wall, 0);
    let mut rng = SplitMix64::new(4);
    let agents = spawn_disk(Vec2::new(-12.0, 0.0), 1_500, 0.5, &mut rng);
    let (mut swarm, _material) = swarm_on(&gpu, &layout, &agents, 2_048);
    for step in 0..600 {
        swarm.step(&gpu, &frame(Vec2::new(12.0, 0.0), step));
    }
    let alive = alive(&swarm, &gpu);
    let inside: Vec<Vec2> = alive
        .iter()
        .map(Agent::pos)
        .filter(|&p| {
            layout
                .index_of(p)
                .is_some_and(|i| KINDS[layout.kinds[i] as usize].flags & FLAG_BLOCKED != 0)
        })
        .collect();
    assert!(
        inside.is_empty(),
        "{} critters inside the wall: {:?}",
        inside.len(),
        &inside[..inside.len().min(5)]
    );
    // The crowd presses against the wall and spills around its ends.
    let crossed = alive.iter().filter(|a| a.pos[0] > 1.0).count();
    assert!(crossed > 0, "no critter got around the wall");
}

#[test]
fn hatchlings_appear_at_the_queen() {
    let Some(gpu) = gpu() else { return };
    let mut rng = SplitMix64::new(6);
    let agents = spawn_disk(Vec2::new(-20.0, 0.0), 50, 0.5, &mut rng);
    let mut swarm = swarm_with(&gpu, &agents, 256);
    let queen = Vec2::new(20.0, 5.0);
    let mut f = frame(Vec2::new(-20.0, 0.0), 0);
    f.hatch = 30;
    f.queen_pos = queen;
    f.queen_radius = 1.3;
    swarm.step(&gpu, &f);
    let counters = swarm.read_counters(&gpu).unwrap();
    assert_eq!(counters[counter::HATCHED], 30);
    assert_eq!(counters[counter::LIVE_END], 80);
    let near_queen = alive(&swarm, &gpu)
        .iter()
        .filter(|a| a.pos().distance(queen) < 4.0)
        .count();
    assert_eq!(near_queen, 30);
}

#[test]
fn hazards_kill_and_dead_slots_are_reused() {
    let Some(gpu) = gpu() else { return };
    let mut rng = SplitMix64::new(8);
    let agents = spawn_disk(Vec2::ZERO, 200, 0.5, &mut rng);
    let mut swarm = swarm_with(&gpu, &agents, 200);
    let mut f = frame(Vec2::ZERO, 0);
    f.hazard = Some(HazardZone {
        center: Vec2::new(2.0, 0.0),
        radius: 2.5,
    });
    swarm.step(&gpu, &f);
    let died = swarm.read_counters(&gpu).unwrap()[counter::DIED];
    assert!(died > 10, "the hazard killed only {died}");
    assert_eq!(alive(&swarm, &gpu).len() as u32, 200 - died);

    // Capacity is full, so new hatchlings must reuse the dead slots.
    let mut f = frame(Vec2::ZERO, 1);
    f.hatch = died + 5;
    f.queen_pos = Vec2::new(-10.0, 0.0);
    swarm.step(&gpu, &f);
    let counters = swarm.read_counters(&gpu).unwrap();
    assert_eq!(
        counters[counter::LIVE_END],
        200,
        "live range must not grow past capacity"
    );
    assert_eq!(
        counters[counter::HATCHED],
        died,
        "only as many as there was room for"
    );
    assert_eq!(alive(&swarm, &gpu).len(), 200);
}

fn session(gpu: &Gpu, hazard: Option<f32>) -> Session {
    let cfg = SessionConfig {
        run: RunConfig {
            initial_critters: 60,
            capacity: 4_096,
            hazard_delay: hazard,
        },
        seed: 1,
    };
    Session::new(gpu, CAPTURE_FORMAT, 16.0 / 9.0, &cfg)
}

/// Steps until the latest feedback has been folded in.
fn settle(gpu: &Gpu, s: &mut Session) {
    gpu.wait_idle();
    s.step(gpu, &Controls::default());
    gpu.wait_idle();
    if let Some(summary) = s.swarm.summary(gpu) {
        s.run.observe(&summary, s.swarm.tuning.rest_density);
    }
}

#[test]
fn session_hatches_banked_mass_and_renders_the_hud() {
    let Some(gpu) = gpu() else { return };
    let mut s = session(&gpu, None);
    s.grant(40.0);
    for _ in 0..120 {
        s.step(&gpu, &Controls::default());
        s.update_camera(1.0 / 60.0);
    }
    settle(&gpu, &mut s);
    assert!(s.run.colony.alive >= 100, "alive {}", s.run.colony.alive);
    let image = headless::capture(&gpu, &s, 320, 180).unwrap();
    // The HUD text is near-white on the top-left.
    let bright = (0..60)
        .flat_map(|x| (0..20).map(move |y| (x, y)))
        .filter(|&(x, y)| image.get_pixel(x, y).0[..3].iter().all(|&c| c > 230))
        .count();
    assert!(bright > 20, "no HUD text drawn");
}

#[test]
fn the_mop_kills_critters_and_can_end_the_run() {
    let Some(gpu) = gpu() else { return };
    let mut s = session(&gpu, Some(0.0));
    // Drag the mop onto the Queen and keep it there.
    let mut lost = false;
    for _ in 0..1_200 {
        let q = s.run.queen.pos;
        if let Some(m) = s.run.mop.as_mut() {
            m.pos = q;
        }
        s.step(&gpu, &Controls::default());
        if s.run.is_over() {
            lost = true;
            break;
        }
    }
    settle(&gpu, &mut s);
    assert!(lost, "the Queen survived a mop on her head for 20 s");
    assert!(s.run.colony.lost > 0, "the mop killed no critters");
}

#[test]
fn headless_scripted_run_grows_the_swarm() {
    if gpu().is_none() {
        return;
    }
    let report = headless::run(&HeadlessOptions {
        frames: 3_600,
        ..Default::default()
    })
    .unwrap();
    eprintln!("{report}");
    assert_eq!(report.stats.non_finite, 0);
    assert!(report.queen_alive);
    assert!(report.peak_mass >= 100, "{report}");
    assert!(report.eaten_fraction > 0.01);
}
