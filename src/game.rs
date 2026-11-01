//! Gameplay state that lives on the CPU: the focal point, Gather/Spread and the
//! camera. Platform-independent (no winit), so headless runs and tests share it.

use glam::Vec2;

use crate::camera::{Camera2D, zoom_for_count};
use crate::render::CameraUniform;
use crate::sim::WorldBounds;

/// Player intent for one step, produced by the input layer or a script.
#[derive(Clone, Copy, Debug, Default)]
pub struct Controls {
    /// Stick/WASD direction; any length, normalised before use.
    pub move_dir: Vec2,
    /// -1 = Spread, 0 = neutral, +1 = Gather.
    pub gather: f32,
    /// When set (mouse held), the focal point heads for this world position.
    pub pointer: Option<Vec2>,
    /// Form slot buttons pressed this frame (edge-triggered).
    pub form: [bool; 2],
    /// Cycle the Form in each slot (edge-triggered; prototype loadout picker).
    pub cycle: [bool; 2],
    /// Build held.
    pub build: bool,
    /// Where the player is aiming, for Forms (world space), if anywhere.
    pub aim: Option<Vec2>,
    /// Start / confirm pressed (edge-triggered).
    pub start: bool,
}

impl Controls {
    /// The same held inputs with this frame's button presses cleared, so a
    /// press fires once even when a frame runs several simulation steps.
    pub fn held(&self) -> Self {
        Self {
            form: [false; 2],
            cycle: [false; 2],
            start: false,
            ..*self
        }
    }
}

#[derive(Clone, Debug)]
pub struct Game {
    pub bounds: WorldBounds,
    pub focal: Vec2,
    /// Smoothed Gather/Spread value sent to the simulation.
    pub gather: f32,
    pub time: f32,
    pub camera: Camera2D,
    /// Critter body length in world units.
    pub agent_size: f32,
    /// Swarm centre from GPU feedback (a frame or two old); `None` until known.
    pub swarm_center: Option<Vec2>,
    /// Radius the swarm occupies at rest.
    pub swarm_radius: f32,
    /// Extra camera pull-back (act transitions).
    pub zoom_scale: f32,
}

/// How far beyond the swarm's edge the focal point may stray. Keeps the swarm
/// from being left behind: the player leads it, never abandons it.
pub const LEASH_MARGIN: f32 = 6.0;

impl Game {
    pub fn new(bounds: WorldBounds, start: Vec2, aspect: f32) -> Self {
        Self {
            bounds,
            focal: start,
            gather: 0.0,
            time: 0.0,
            camera: Camera2D::new(start, zoom_for_count(0), aspect),
            agent_size: 0.45,
            swarm_center: None,
            swarm_radius: 0.0,
            zoom_scale: 1.0,
        }
    }

    /// Top critter speed in world units/s. Scales with the view, so the swarm
    /// crosses the screen at the same pace whether it is ten critters or 30k.
    pub fn swarm_speed(&self) -> f32 {
        (self.camera.half_height * 0.45).clamp(8.0, 28.0)
    }

    /// How fast the focal point moves: just under the swarm's speed, so the
    /// swarm can keep up with it.
    pub fn focal_speed(&self) -> f32 {
        self.swarm_speed() * 0.9
    }

    /// Advances one fixed simulation step.
    pub fn update(&mut self, controls: &Controls, dt: f32) {
        self.time += dt;
        let speed = self.focal_speed();
        if let Some(target) = controls.pointer {
            let to = target - self.focal;
            self.focal += to.clamp_length_max(speed * dt);
        } else {
            self.focal += controls.move_dir.normalize_or_zero() * speed * dt;
        }
        if let Some(center) = self.swarm_center {
            let leash = self.swarm_radius + LEASH_MARGIN;
            self.focal = center + (self.focal - center).clamp_length_max(leash);
        }
        self.focal = self.focal.clamp(self.bounds.min, self.bounds.max);
        let target = controls.gather.clamp(-1.0, 1.0);
        let max_delta = 6.0 * dt;
        self.gather += (target - self.gather).clamp(-max_delta, max_delta);
    }

    /// Records the latest swarm feedback.
    pub fn observe_swarm(&mut self, center: Vec2, radius: f32) {
        self.swarm_center = Some(center);
        self.swarm_radius = radius;
    }

    /// Where the camera aims: mostly at the swarm, leaning toward the focal
    /// point so the player sees where they are heading.
    pub fn camera_target(&self) -> Vec2 {
        match self.swarm_center {
            Some(center) => center.lerp(self.focal, 0.3),
            None => self.focal,
        }
    }

    /// Eases the camera toward its target and the zoom for `count` critters.
    pub fn update_camera(&mut self, count: u32, dt: f32) {
        self.camera.follow(
            self.camera_target(),
            zoom_for_count(count) * self.zoom_scale,
            dt,
        );
    }

    pub fn camera_uniform(
        &self,
        viewport_height: u32,
        mat_dim: [u32; 2],
        mat_cell: f32,
    ) -> CameraUniform {
        CameraUniform {
            center: self.camera.center.to_array(),
            half_extent: self.camera.half_extent().to_array(),
            focal: self.focal.to_array(),
            time: self.time,
            agent_size: self.agent_size,
            world_min: self.bounds.min.to_array(),
            world_max: self.bounds.max.to_array(),
            gather: self.gather,
            pixel_world: self.camera.pixel_world(viewport_height),
            mat_dim,
            mat_cell,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn game() -> Game {
        let bounds = WorldBounds {
            min: Vec2::splat(-50.0),
            max: Vec2::splat(50.0),
        };
        Game::new(bounds, Vec2::ZERO, 16.0 / 9.0)
    }

    #[test]
    fn focal_moves_with_input_and_stays_in_bounds() {
        let mut g = game();
        let c = Controls {
            move_dir: Vec2::new(1.0, 0.0),
            ..Default::default()
        };
        g.update(&c, 0.5);
        assert!(g.focal.x > 0.0 && g.focal.y == 0.0);
        for _ in 0..1000 {
            g.update(&c, 0.1);
        }
        assert_eq!(g.focal.x, 50.0);
    }

    #[test]
    fn pointer_overrides_keys() {
        let mut g = game();
        let c = Controls {
            move_dir: Vec2::new(1.0, 0.0),
            pointer: Some(Vec2::new(0.0, -10.0)),
            ..Default::default()
        };
        for _ in 0..200 {
            g.update(&c, 1.0 / 60.0);
        }
        assert!((g.focal - Vec2::new(0.0, -10.0)).length() < 1e-3);
    }

    #[test]
    fn focal_is_leashed_to_the_swarm() {
        let mut g = game();
        g.observe_swarm(Vec2::new(-20.0, 0.0), 4.0);
        let c = Controls {
            move_dir: Vec2::new(1.0, 0.0),
            ..Default::default()
        };
        for _ in 0..600 {
            g.update(&c, 1.0 / 60.0);
        }
        assert!(
            (g.focal.x - (-20.0 + 4.0 + LEASH_MARGIN)).abs() < 1e-3,
            "{}",
            g.focal
        );
        // The camera frames the swarm, leaning toward the focal point.
        let t = g.camera_target();
        assert!(t.x > -20.0 && t.x < g.focal.x);
    }

    #[test]
    fn gather_is_smoothed() {
        let mut g = game();
        let c = Controls {
            gather: 1.0,
            ..Default::default()
        };
        g.update(&c, 1.0 / 60.0);
        assert!(g.gather > 0.0 && g.gather < 1.0);
        for _ in 0..60 {
            g.update(&c, 1.0 / 60.0);
        }
        assert_eq!(g.gather, 1.0);
    }
}
