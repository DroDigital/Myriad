//! Windowless runs: a scripted player, timing, stats and an optional PNG.
//! Used for benchmarks, CI smoke tests and screenshots.

use std::fmt;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use glam::Vec2;

use crate::game::Controls;
use crate::gpu::Gpu;
use crate::level;
use crate::run::{Run, RunConfig, RunState};
use crate::session::{SIM_DT, Session, SessionConfig};
use crate::sim::stats::SwarmStats;

#[derive(Clone, Debug)]
pub struct HeadlessOptions {
    pub agents: u32,
    pub capacity: u32,
    pub frames: u32,
    pub width: u32,
    pub height: u32,
    pub seed: u64,
    /// Include the mop hazard (off by default so scripted runs are stable).
    pub hazard: bool,
    pub screenshot: Option<PathBuf>,
}

impl Default for HeadlessOptions {
    fn default() -> Self {
        Self {
            agents: 10,
            capacity: 32_768,
            frames: 3_600,
            width: 1280,
            height: 720,
            seed: 1,
            hazard: false,
            screenshot: None,
        }
    }
}

#[derive(Debug)]
pub struct HeadlessReport {
    pub adapter: String,
    pub frames: u32,
    pub ms_per_step: f64,
    pub stats: SwarmStats,
    pub mass: u32,
    pub peak_mass: u32,
    pub eaten_fraction: f64,
    pub queen_alive: bool,
    /// Share of living critters in the main body (stragglers drag it down).
    pub body_share: f64,
    pub screenshot: Option<PathBuf>,
}

impl fmt::Display for HeadlessReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "adapter: {}", self.adapter)?;
        writeln!(f, "{} steps: {:.3} ms/step", self.frames, self.ms_per_step)?;
        writeln!(
            f,
            "run: mass {} (peak {}) | {:.1}% of the kitchen eaten | queen {} | {:.1}% in the main body",
            self.mass,
            self.peak_mass,
            self.eaten_fraction * 100.0,
            if self.queen_alive { "alive" } else { "fallen" },
            self.body_share * 100.0
        )?;
        write!(f, "swarm: {}", self.stats)?;
        if let Some(path) = &self.screenshot {
            write!(f, "\nscreenshot: {}", path.display())?;
        }
        Ok(())
    }
}

/// A scripted player: tours the level's food. At each spot it waits for the
/// swarm to arrive, eats until the food runs dry, then moves on.
pub struct Script {
    tour: Vec<Vec2>,
    index: usize,
    /// Seconds spent at the current spot since arriving.
    dwell: f32,
    /// Seconds since setting off for the current spot.
    travel: f32,
    /// Food eaten when the last progress check happened, and when that was.
    last_food: f64,
    since_progress: f32,
}

impl Script {
    /// Leave once nothing new has been eaten for this long.
    const STARVED: f32 = 2.5;
    /// Give up on a spot after this long, arrived or not.
    const MAX_TIME: f32 = 30.0;

    pub fn new(start: Vec2, food: &[Vec2]) -> Self {
        Self {
            tour: level::tour(start, food),
            index: 0,
            dwell: 0.0,
            travel: 0.0,
            last_food: 0.0,
            since_progress: 0.0,
        }
    }

    pub fn target(&self) -> Option<Vec2> {
        self.tour.get(self.index).copied()
    }

    pub fn controls(&mut self, run: &Run, dt: f32) -> Controls {
        let Some(target) = self.target() else {
            return Controls::default();
        };
        self.travel += dt;
        let arrived = run
            .game
            .swarm_center
            .is_some_and(|c| c.distance(target) < run.game.swarm_radius + 4.0);
        if arrived {
            self.dwell += dt;
        }
        let food = run.colony.food_eaten;
        if food > self.last_food + 0.5 {
            self.last_food = food;
            self.since_progress = 0.0;
        } else {
            self.since_progress += dt;
        }
        let done = (arrived && self.dwell > 2.0 && self.since_progress > Self::STARVED)
            || self.travel > Self::MAX_TIME;
        if done {
            self.index = (self.index + 1) % self.tour.len();
            self.dwell = 0.0;
            self.travel = 0.0;
            self.since_progress = 0.0;
        }
        Controls {
            pointer: Some(target),
            // Stuck on the way (water in the way, most likely): build.
            build: !arrived && self.travel > 4.0,
            ..Default::default()
        }
    }
}

/// Share of living critters within reach of the focal point (the swarm's
/// main body: the swarm's rest radius plus a margin).
pub fn body_share(agents: &[crate::sim::Agent], focal: Vec2, swarm_radius: f32) -> f64 {
    let reach = swarm_radius * 2.0 + 8.0;
    let alive: Vec<_> = agents.iter().filter(|a| a.is_alive()).collect();
    if alive.is_empty() {
        return 0.0;
    }
    let near = alive
        .iter()
        .filter(|a| a.pos().distance(focal) < reach)
        .count();
    near as f64 / alive.len() as f64
}

pub const CAPTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

pub fn run(opts: &HeadlessOptions) -> Result<HeadlessReport> {
    let gpu = Gpu::headless()?;
    log::info!("adapter: {}", gpu.adapter_summary());
    let cfg = SessionConfig {
        run: RunConfig {
            initial_critters: opts.agents,
            capacity: opts.capacity.max(opts.agents),
            hazard_delay: opts.hazard.then_some(20.0),
        },
        seed: opts.seed,
    };
    let aspect = opts.width as f32 / opts.height.max(1) as f32;
    let mut session = Session::new(&gpu, CAPTURE_FORMAT, aspect, &cfg);
    let mut script = Script::new(session.level.queen_start, &session.level.food_spots);

    gpu.wait_idle();
    let start = Instant::now();
    for _ in 0..opts.frames {
        let controls = script.controls(&session.run, SIM_DT);
        session.step(&gpu, &controls);
        if session.swarm.steps().is_multiple_of(300) {
            let r = &session.run;
            log::debug!(
                "t={:.0}s target={:?} center={:?} focal={} food={:.1} mass={} queen={}",
                r.elapsed,
                script.target(),
                r.game.swarm_center,
                r.game.focal,
                r.colony.food_eaten,
                r.colony.mass(),
                r.queen.pos,
            );
        }
        session.update_camera(SIM_DT);
    }
    gpu.wait_idle();
    let elapsed = start.elapsed().as_secs_f64();
    // Let the last readbacks land so the report is current.
    session.step(&gpu, &Controls::default());
    gpu.wait_idle();
    if let Some(s) = session.swarm.summary(&gpu) {
        session.run.observe(&s, session.swarm.tuning.rest_density);
    }

    let agents = session.swarm.read_agents(&gpu)?;
    let stats = SwarmStats::compute(&agents, session.run.game.focal, session.swarm.bounds());

    let screenshot = match &opts.screenshot {
        Some(path) => {
            let image = capture(&gpu, &session, opts.width, opts.height)?;
            if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
                std::fs::create_dir_all(dir)?;
            }
            image
                .save(path)
                .with_context(|| format!("saving {}", path.display()))?;
            Some(path.clone())
        }
        None => None,
    };

    let run = &session.run;
    Ok(HeadlessReport {
        adapter: gpu.adapter_summary(),
        frames: opts.frames,
        ms_per_step: if opts.frames > 0 {
            elapsed * 1000.0 / opts.frames as f64
        } else {
            0.0
        },
        stats,
        mass: run.colony.mass(),
        peak_mass: run.colony.peak_mass,
        eaten_fraction: run.eaten_fraction(),
        queen_alive: run.state == RunState::Playing,
        body_share: body_share(&agents, run.game.focal, run.game.swarm_radius),
        screenshot,
    })
}

pub fn capture(gpu: &Gpu, session: &Session, width: u32, height: u32) -> Result<image::RgbaImage> {
    assert_eq!(session.renderer.format(), CAPTURE_FORMAT);
    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("capture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: CAPTURE_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    session.draw(gpu, &view, width, height);

    let row = width * 4;
    let padded_row =
        row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("capture staging"),
        size: padded_row as u64 * height as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("capture copy"),
        });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit([encoder.finish()]);
    let bytes: Vec<u8> = gpu.map_and_read(&staging)?;
    let mut pixels = Vec::with_capacity((row * height) as usize);
    for y in 0..height as usize {
        let start = y * padded_row as usize;
        pixels.extend_from_slice(&bytes[start..start + row as usize]);
    }
    image::RgbaImage::from_raw(width, height, pixels).context("capture buffer size mismatch")
}
