//! One running game: GPU swarm + material + renderer + the run's rules,
//! driven at a fixed timestep. Shared by the window app and headless runs.

use crate::game::Controls;
use crate::gpu::Gpu;
use crate::hud;
use crate::level::Level;
use crate::queen::QueenState;
use crate::render::{FrameData, OverlayUniform, Renderer};
use crate::run::{Run, RunConfig};
use crate::sim::material::MaterialGrid;
use crate::sim::{Swarm, SwarmConfig, spawn_disk};
use crate::util::SplitMix64;

/// Simulation rate. Rendering may run faster or slower.
pub const SIM_DT: f32 = 1.0 / 60.0;
/// Most steps run per frame before the backlog is dropped (avoids spiralling).
pub const MAX_STEPS_PER_FRAME: u32 = 4;
/// Distance between critters at spawn.
const SPAWN_SPACING: f32 = 0.5;

#[derive(Clone, Copy, Debug, Default)]
pub struct SessionConfig {
    pub run: RunConfig,
    pub seed: u64,
}

pub struct Session {
    pub level: Level,
    pub material: MaterialGrid,
    pub swarm: Swarm,
    pub renderer: Renderer,
    pub run: Run,
    cfg: SessionConfig,
    aspect: f32,
    /// What the GPU was last told about threats, for drawing.
    last_threats: [crate::sim::Threat; crate::sim::MAX_THREATS],
    accumulator: f32,
}

impl Session {
    pub fn new(gpu: &Gpu, format: wgpu::TextureFormat, aspect: f32, cfg: &SessionConfig) -> Self {
        let level = Level::kitchen(cfg.seed);
        let material = MaterialGrid::new(gpu, &level.material);
        let mut swarm = Swarm::new(
            gpu,
            SwarmConfig {
                capacity: cfg.run.capacity,
                bounds: level.bounds(),
                cell_size: 1.0,
            },
            &material,
        );
        let mut rng = SplitMix64::new(cfg.seed);
        let initial = cfg.run.initial_critters.min(cfg.run.capacity) as usize;
        swarm.spawn(
            &gpu.queue,
            &spawn_disk(level.queen_start, initial, SPAWN_SPACING, &mut rng),
        );
        let renderer = Renderer::new(
            gpu,
            format,
            swarm.agents_buffer(),
            swarm.capacity(),
            &material,
        );
        let run = Run::new(&level, &cfg.run, aspect);
        Self {
            level,
            material,
            swarm,
            renderer,
            run,
            cfg: *cfg,
            aspect,
            last_threats: Default::default(),
            accumulator: 0.0,
        }
    }

    /// Starts a fresh run with the same settings.
    pub fn restart(&mut self, gpu: &Gpu) {
        let aspect = self.run.game.camera.aspect;
        *self = Session::new(gpu, self.renderer.format(), aspect, &self.cfg);
        self.aspect = aspect;
    }

    /// Runs exactly one fixed simulation step.
    pub fn step(&mut self, gpu: &Gpu, controls: &Controls) {
        let frame = self.run.update(controls, SIM_DT);
        self.last_threats = frame.threats;
        self.swarm.tuning.max_speed = self.run.game.swarm_speed();
        self.swarm.step(gpu, &frame);
        if let Some(summary) = self.swarm.summary(gpu) {
            self.run.observe(&summary, self.swarm.tuning.rest_density);
        }
    }

    /// Advances by a variable frame time using fixed steps. Returns steps run.
    pub fn advance(&mut self, gpu: &Gpu, controls: &Controls, frame_dt: f32) -> u32 {
        self.accumulator += frame_dt.clamp(0.0, 0.25);
        let mut steps = 0;
        let held = controls.held();
        while self.accumulator >= SIM_DT && steps < MAX_STEPS_PER_FRAME {
            // Button presses fire on the first step of the frame only.
            self.step(gpu, if steps == 0 { controls } else { &held });
            self.accumulator -= SIM_DT;
            steps += 1;
        }
        if steps == MAX_STEPS_PER_FRAME {
            self.accumulator = self.accumulator.min(SIM_DT);
        }
        self.update_camera(frame_dt);
        steps
    }

    pub fn update_camera(&mut self, dt: f32) {
        self.run.game.update_camera(self.run.colony.alive, dt);
    }

    /// Adds mass directly (developer cheat).
    pub fn grant(&mut self, critters: f64) {
        self.run.colony.grant(critters);
    }

    pub fn overlay(&self) -> OverlayUniform {
        let q = &self.run.queen;
        OverlayUniform {
            queen_pos: q.pos.to_array(),
            queen_heading: q.heading.to_array(),
            queen_radius: crate::queen::QUEEN_RADIUS,
            queen_health: q.health / crate::queen::MAX_HEALTH,
            queen_flash: q.hurt_flash,
            queen_state: match q.state {
                QueenState::Healthy => 0.0,
                QueenState::LastStand { .. } => 1.0,
                QueenState::Dead => 2.0,
            },
            hazard_pos: [0.0; 2],
            hazard_radius: 0.0,
            hazard_state: 0.0,
            hazard_warn: 0.0,
            guard_radius: self.swarm.tuning.guard_radius,
            _pad: [0.0; 2],
        }
    }

    pub fn draw(&self, gpu: &Gpu, target: &wgpu::TextureView, width: u32, height: u32) {
        let text = hud::build(&self.run, width as f32, height as f32);
        let frame = FrameData {
            camera: self
                .run
                .game
                .camera_uniform(height, self.material.dim, self.material.cell),
            overlay: self.overlay(),
            glyphs: &text.glyphs,
            viewport: [width, height],
            threats: self.last_threats,
        };
        self.renderer.draw(gpu, target, &frame);
    }
}
