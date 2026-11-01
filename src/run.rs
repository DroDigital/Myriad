//! One run's rules on the CPU: steering, the Queen, the mop, the mass
//! economy, and win/lose. Produces the [`SimFrame`] for each GPU step and
//! consumes [`SwarmSummary`] feedback. No GPU code, so it is unit-testable.

use crate::colony::Colony;
use crate::game::{Controls, Game};
use crate::hazard::{MOP_QUEEN_DPS, Mop};
use crate::level::Level;
use crate::queen::{QUEEN_RADIUS, Queen, SPEED_FACTOR, guard_factor};
use crate::sim::feedback::SwarmSummary;
use crate::sim::material::MaterialLayout;
use crate::sim::{SimFrame, rest_radius};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RunState {
    Playing,
    /// The Queen fell. The run is over.
    Lost,
}

#[derive(Clone, Copy, Debug)]
pub struct RunConfig {
    pub initial_critters: u32,
    pub capacity: u32,
    /// Seconds before the mop appears; `None` for no mop.
    pub hazard_delay: Option<f32>,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            initial_critters: 10,
            capacity: 32_768,
            hazard_delay: Some(30.0),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Run {
    pub game: Game,
    pub queen: Queen,
    pub mop: Option<Mop>,
    pub colony: Colony,
    pub state: RunState,
    pub elapsed: f32,
    /// Critters guarding the Queen, per the latest summary.
    pub guards: u32,
    /// Food in the level at the start, in full cells.
    pub food_total: f64,
    capacity: u32,
    walls: MaterialLayout,
}

impl Run {
    pub fn new(level: &Level, cfg: &RunConfig, aspect: f32) -> Self {
        let start = level.queen_start;
        let mut game = Game::new(level.bounds(), start, aspect);
        game.camera.half_height = crate::camera::zoom_for_count(cfg.initial_critters);
        Self {
            game,
            queen: Queen::new(start),
            mop: cfg
                .hazard_delay
                .map(|delay| Mop::new(level.hazard_route.clone(), delay)),
            colony: Colony::new(cfg.initial_critters),
            state: RunState::Playing,
            elapsed: 0.0,
            guards: 0,
            food_total: level.material.total_food() as f64 / crate::sim::material::FULL as f64,
            capacity: cfg.capacity,
            walls: level.material.clone(),
        }
    }

    pub fn is_over(&self) -> bool {
        self.state != RunState::Playing
    }

    /// Share of the level's food eaten, 0..1.
    pub fn eaten_fraction(&self) -> f64 {
        if self.food_total > 0.0 {
            (self.colony.food_eaten / self.food_total).min(1.0)
        } else {
            0.0
        }
    }

    /// Folds in GPU feedback.
    pub fn observe(&mut self, s: &SwarmSummary, rest_density: f32) {
        self.colony.observe(s);
        self.guards = s.near_queen;
        if s.body > 0 {
            self.game
                .observe_swarm(s.centroid, rest_radius(s.body, rest_density));
        }
    }

    /// Advances one fixed step of rules and returns what the GPU should do.
    pub fn update(&mut self, controls: &Controls, dt: f32) -> SimFrame {
        let playing = self.state == RunState::Playing;
        if playing {
            self.elapsed += dt;
            self.game.update(controls, dt);
        } else {
            // The swarm mills about the fallen Queen.
            self.game.update(&Controls::default(), dt);
        }
        let walls = &self.walls;
        self.queen.update(
            self.game.swarm_center,
            self.game.swarm_speed() * SPEED_FACTOR,
            dt,
            |p| walls.is_wall(p),
        );
        if let Some(mop) = &mut self.mop {
            mop.update(dt);
            if playing && mop.touches(self.queen.pos, QUEEN_RADIUS) {
                self.queen
                    .damage(MOP_QUEEN_DPS * guard_factor(self.guards) * dt);
            }
        }
        if playing && self.queen.is_dead() {
            self.state = RunState::Lost;
        }
        let hatch = if self.state == RunState::Playing {
            self.colony.take_hatch(self.capacity)
        } else {
            0
        };
        SimFrame {
            focal: if self.is_over() {
                self.queen.pos
            } else {
                self.game.focal
            },
            gather: self.game.gather,
            dt,
            time: self.game.time,
            hatch,
            queen_pos: self.queen.pos,
            queen_radius: QUEEN_RADIUS,
            hazard: self.mop.as_ref().and_then(Mop::zone),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::queen::MAX_HEALTH;

    const DT: f32 = 1.0 / 60.0;

    fn run(hazard: Option<f32>) -> Run {
        let cfg = RunConfig {
            hazard_delay: hazard,
            ..Default::default()
        };
        Run::new(&Level::kitchen(1), &cfg, 16.0 / 9.0)
    }

    #[test]
    fn eating_feeds_hatching_at_the_queen() {
        let mut r = run(None);
        r.observe(
            &SwarmSummary {
                alive: 10,
                body: 10,
                eaten_total: 10 * crate::sim::material::FULL,
                centroid: r.queen.pos,
                ..Default::default()
            },
            2.5,
        );
        let frame = r.update(&Controls::default(), DT);
        assert_eq!(frame.hatch, 10, "10 cells of food -> 10 critters");
        assert_eq!(frame.queen_pos, r.queen.pos);
    }

    #[test]
    fn queen_death_ends_the_run() {
        let mut r = run(Some(0.0));
        // Park the Queen on the mop with no guards.
        let mop_pos = r.mop.as_ref().unwrap().pos;
        r.queen.pos = mop_pos;
        r.queen.damage(MAX_HEALTH - 1.0);
        for _ in 0..600 {
            r.queen.pos = r.mop.as_ref().unwrap().pos;
            r.update(&Controls::default(), DT);
            if r.is_over() {
                break;
            }
        }
        assert_eq!(r.state, RunState::Lost);
        // No more hatching after death, even with food in the bank.
        r.colony.grant(100.0);
        assert_eq!(r.update(&Controls::default(), DT).hatch, 0);
    }

    #[test]
    fn guards_slow_the_damage() {
        let mut exposed = run(Some(0.0));
        let mut guarded = run(Some(0.0));
        guarded.guards = 60;
        for r in [&mut exposed, &mut guarded] {
            for _ in 0..60 {
                r.queen.pos = r.mop.as_ref().unwrap().pos;
                r.update(&Controls::default(), DT);
            }
        }
        assert!(guarded.queen.health > exposed.queen.health);
    }
}
