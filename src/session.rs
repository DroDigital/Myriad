//! One running game: swarm + renderer + gameplay state, driven at a fixed
//! timestep. Shared by the window app and headless runs.

use glam::Vec2;

use crate::game::{Controls, Game};
use crate::gpu::Gpu;
use crate::render::Renderer;
use crate::sim::{Swarm, SwarmConfig, WorldBounds, rest_radius, spawn_disk};
use crate::util::SplitMix64;

/// Simulation rate. Rendering may run faster or slower.
pub const SIM_DT: f32 = 1.0 / 60.0;
/// Most steps run per frame before the backlog is dropped (avoids spiralling).
pub const MAX_STEPS_PER_FRAME: u32 = 4;
/// Distance between critters at spawn.
const SPAWN_SPACING: f32 = 0.5;

#[derive(Clone, Copy, Debug)]
pub struct SessionConfig {
    pub initial_agents: u32,
    pub capacity: u32,
    pub seed: u64,
    pub bounds: WorldBounds,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            initial_agents: 2_000,
            capacity: 32_768,
            seed: 1,
            // Prototype kitchen floor, in world units (one critter ~ 0.45).
            bounds: WorldBounds {
                min: Vec2::new(-120.0, -80.0),
                max: Vec2::new(120.0, 80.0),
            },
        }
    }
}

pub struct Session {
    pub swarm: Swarm,
    pub renderer: Renderer,
    pub game: Game,
    rng: SplitMix64,
    accumulator: f32,
}

impl Session {
    pub fn new(gpu: &Gpu, format: wgpu::TextureFormat, aspect: f32, cfg: &SessionConfig) -> Self {
        let mut swarm = Swarm::new(
            &gpu.device,
            SwarmConfig {
                capacity: cfg.capacity,
                bounds: cfg.bounds,
                cell_size: 1.0,
            },
        );
        let renderer = Renderer::new(&gpu.device, format, swarm.agents_buffer());
        let start = (cfg.bounds.min + cfg.bounds.max) * 0.5;
        let mut game = Game::new(cfg.bounds, start, aspect);
        let mut rng = SplitMix64::new(cfg.seed);
        let initial = cfg.initial_agents.min(cfg.capacity) as usize;
        swarm.spawn(
            &gpu.queue,
            &spawn_disk(start, initial, SPAWN_SPACING, &mut rng),
        );
        // Start already zoomed to fit, rather than easing out from the minimum.
        game.camera.half_height = crate::camera::zoom_for_count(swarm.count());
        Self {
            swarm,
            renderer,
            game,
            rng,
            accumulator: 0.0,
        }
    }

    /// Runs exactly one fixed simulation step.
    pub fn step(&mut self, gpu: &Gpu, controls: &Controls) {
        self.game.update(controls, SIM_DT);
        self.swarm.tuning.max_speed = self.game.swarm_speed();
        self.swarm.step(gpu, &self.game.sim_frame(SIM_DT));
        if let Some(summary) = self.swarm.summary(gpu) {
            let radius = rest_radius(summary.alive, self.swarm.tuning.rest_density);
            self.game.observe_swarm(summary.centroid, radius);
        }
    }

    /// Advances by a variable frame time using fixed steps. Returns steps run.
    pub fn advance(&mut self, gpu: &Gpu, controls: &Controls, frame_dt: f32) -> u32 {
        self.accumulator += frame_dt.clamp(0.0, 0.25);
        let mut steps = 0;
        while self.accumulator >= SIM_DT && steps < MAX_STEPS_PER_FRAME {
            self.step(gpu, controls);
            self.accumulator -= SIM_DT;
            steps += 1;
        }
        if steps == MAX_STEPS_PER_FRAME {
            self.accumulator = self.accumulator.min(SIM_DT);
        }
        self.game.update_camera(self.swarm.count(), frame_dt);
        steps
    }

    /// Hatches up to `n` new critters around the focal point.
    pub fn hatch(&mut self, gpu: &Gpu, n: u32) -> usize {
        let room = self.swarm.capacity() - self.swarm.count();
        let n = n.min(room) as usize;
        let agents = spawn_disk(self.game.focal, n, SPAWN_SPACING, &mut self.rng);
        self.swarm.spawn(&gpu.queue, &agents)
    }

    pub fn draw(&self, gpu: &Gpu, target: &wgpu::TextureView, viewport_height: u32) {
        let camera = self.game.camera_uniform(viewport_height);
        self.renderer.draw(gpu, target, &camera, self.swarm.count());
    }
}
