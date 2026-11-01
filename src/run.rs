//! One run's rules on the CPU: steering, the Queen, Forms, threats, the mass
//! economy, and the act's arc from the title to the Cleaner and beyond.
//! Produces the [`SimFrame`] for each GPU step and consumes [`SwarmSummary`]
//! feedback. No GPU code, so it is unit-testable.

use glam::Vec2;

use crate::colony::Colony;
use crate::forms::{FormKind, Forms};
use crate::game::{Controls, Game};
use crate::level::Level;
use crate::queen::{QUEEN_RADIUS, Queen, SPEED_FACTOR, guard_factor};
use crate::sim::feedback::SwarmSummary;
use crate::sim::material::MaterialLayout;
use crate::sim::{FormShape, MAX_THREATS, SimFrame, Threat, rest_radius};
use crate::threats::{Enemy, EnemyKind, Response, spawn_point};
use crate::util::SplitMix64;

/// Share of damage the Queen takes while the Shield is held.
pub const SHIELD_DAMAGE_FACTOR: f32 = 0.15;
/// Seconds the act transition lasts before the act is won.
pub const TRANSITION_SECS: f32 = 4.0;
/// Seconds a narrator line stays up.
const NARRATION_SECS: f32 = 4.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RunState {
    /// Waiting on the title screen.
    Title,
    Playing,
    /// The Cleaner is beaten: the camera pulls back and the swarm consolidates.
    Transition {
        t: f32,
    },
    /// The act is cleared (the end of the prototype).
    Won,
    /// The Queen fell. The run is over.
    Lost,
}

#[derive(Clone, Copy, Debug)]
pub struct RunConfig {
    pub initial_critters: u32,
    pub capacity: u32,
    /// Spawn threats and the boss.
    pub threats: bool,
    /// Start on the title screen (the window app); otherwise start playing.
    pub title: bool,
    /// Mass at which the Cleaner arrives (it also comes at half the kitchen
    /// eaten). 0 brings it at once.
    pub boss_mass: u32,
    /// Multiplies the Cleaner's health.
    pub boss_health: f32,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            initial_critters: 10,
            capacity: 32_768,
            threats: true,
            title: false,
            boss_mass: 800,
            boss_health: 1.0,
        }
    }
}

/// Summary for the end screens.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RunSummary {
    pub seconds: f32,
    pub peak_mass: u32,
    pub eaten_fraction: f64,
}

#[derive(Clone, Debug)]
pub struct Run {
    pub game: Game,
    pub queen: Queen,
    pub colony: Colony,
    pub state: RunState,
    pub elapsed: f32,
    /// Critters guarding the Queen, per the latest summary.
    pub guards: u32,
    /// Food in the level at the start, in full cells.
    pub food_total: f64,
    pub forms: Forms,
    /// Build is held this step.
    pub building: bool,
    pub enemies: Vec<Enemy>,
    pub response: Response,
    /// The Cleaner has arrived (it may since have been beaten).
    pub boss_arrived: bool,
    /// Current narrator line and seconds left on screen.
    pub narration: Option<(&'static str, f32)>,
    cfg: RunConfig,
    walls: MaterialLayout,
    rng: SplitMix64,
    /// Latest per-slot contact-damage counters.
    hits: [u32; MAX_THREATS],
    merge_requested: bool,
}

impl Run {
    pub fn new(level: &Level, cfg: &RunConfig, aspect: f32) -> Self {
        let start = level.queen_start;
        let mut game = Game::new(level.bounds(), start, aspect);
        game.camera.half_height = crate::camera::zoom_for_count(cfg.initial_critters);
        Self {
            game,
            queen: Queen::new(start),
            colony: Colony::new(cfg.initial_critters),
            state: if cfg.title {
                RunState::Title
            } else {
                RunState::Playing
            },
            elapsed: 0.0,
            guards: 0,
            food_total: level.material.total_food() as f64 / crate::sim::material::FULL as f64,
            forms: Forms::default(),
            building: false,
            enemies: Vec::new(),
            response: Response::default(),
            boss_arrived: false,
            narration: None,
            cfg: *cfg,
            walls: level.material.clone(),
            rng: SplitMix64::new(0x5eed),
            hits: [0; MAX_THREATS],
            merge_requested: false,
        }
    }

    pub fn is_over(&self) -> bool {
        matches!(self.state, RunState::Won | RunState::Lost)
    }

    pub fn is_playing(&self) -> bool {
        self.state == RunState::Playing
    }

    /// Share of the level's food eaten, 0..1.
    pub fn eaten_fraction(&self) -> f64 {
        if self.food_total > 0.0 {
            (self.colony.food_eaten / self.food_total).min(1.0)
        } else {
            0.0
        }
    }

    pub fn summary(&self) -> RunSummary {
        RunSummary {
            seconds: self.elapsed,
            peak_mass: self.colony.peak_mass,
            eaten_fraction: self.eaten_fraction(),
        }
    }

    /// The Cleaner, if it is on the board.
    pub fn boss(&self) -> Option<&Enemy> {
        self.enemies.iter().find(|e| e.kind == EnemyKind::Cleaner)
    }

    fn narrate(&mut self, line: &'static str) {
        self.narration = Some((line, NARRATION_SECS));
    }

    /// Folds in GPU feedback.
    pub fn observe(&mut self, s: &SwarmSummary, rest_density: f32) {
        self.colony.observe(s);
        self.guards = s.near_queen;
        self.hits = s.enemy_hits;
        for e in &mut self.enemies {
            e.observe_hits(&s.enemy_hits);
        }
        if s.body > 0 {
            self.game
                .observe_swarm(s.centroid, rest_radius(s.body, rest_density));
        }
    }

    /// Advances one fixed step of rules and returns what the GPU should do.
    pub fn update(&mut self, controls: &Controls, dt: f32) -> SimFrame {
        if self.state == RunState::Title && controls.start {
            self.state = RunState::Playing;
        }
        let playing = self.is_playing();
        if let Some((_, t)) = &mut self.narration {
            *t -= dt;
            if *t <= 0.0 {
                self.narration = None;
            }
        }
        if playing {
            self.elapsed += dt;
            self.game.update(controls, dt);
        } else {
            // On the title and end screens the swarm mills about the Queen.
            self.game.update(&Controls::default(), dt);
        }
        let walls = &self.walls;
        self.queen.update(
            self.game.swarm_center,
            self.game.swarm_speed() * SPEED_FACTOR,
            dt,
            |p| walls.is_wall(p),
        );

        // Forms: cycle loadout, start one on a press, then hold it.
        let mut sacrifice = 0;
        if playing {
            for slot in 0..2 {
                if controls.cycle[slot] {
                    self.forms.cycle(slot);
                }
                if controls.form[slot] {
                    let aim = self.aim(controls);
                    if let Ok(cost) = self.forms.try_activate(slot, self.colony.mass(), aim) {
                        sacrifice += self.colony.spend(cost);
                        match self.forms.slots[slot] {
                            FormKind::Shield => self.response.shield_uses += 1,
                            FormKind::Spear => self.response.spear_uses += 1,
                            _ => {}
                        }
                    }
                }
            }
        }
        self.forms.update(dt);
        self.building = playing && controls.build;
        let form = self.form_shape();
        // Spear and Comet carry the swarm along their aim.
        if let Some(f) = form
            && (f.kind == FormKind::Spear as u32 || f.kind == FormKind::Comet as u32)
        {
            self.game.focal = (f.origin + f.dir * f.size.max(6.0))
                .clamp(self.game.bounds.min, self.game.bounds.max);
        }

        if playing && self.cfg.threats {
            self.update_threats(dt, form);
        }
        if playing && self.queen.is_dead() {
            self.state = RunState::Lost;
        }

        let mut merge = false;
        if let RunState::Transition { t } = self.state {
            if !self.merge_requested {
                self.merge_requested = true;
                merge = true;
            }
            let t = t + dt;
            self.state = if t >= TRANSITION_SECS {
                RunState::Won
            } else {
                RunState::Transition { t }
            };
        }
        // The act transition pulls the camera back.
        self.game.zoom_scale = match self.state {
            RunState::Transition { t } => 1.0 + 0.6 * (t / TRANSITION_SECS).min(1.0),
            RunState::Won => 1.6,
            _ => 1.0,
        };

        let hatch = if playing {
            self.colony.take_hatch(self.cfg.capacity)
        } else {
            0
        };
        SimFrame {
            focal: if self.is_over() || self.state == RunState::Title {
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
            hazard: None,
            form,
            build: self.building,
            sacrifice,
            threats: self.threat_uniforms(),
            merge,
        }
    }

    fn update_threats(&mut self, dt: f32, form: Option<FormShape>) {
        // The Response level rises with the swarm.
        if let Some(line) = self.response.update_level(self.colony.mass(), self.elapsed) {
            self.narrate(line);
        }
        // The Cleaner arrives once the swarm is big or the kitchen half eaten.
        if !self.boss_arrived
            && (self.colony.mass() >= self.cfg.boss_mass || self.eaten_fraction() >= 0.5)
        {
            self.boss_arrived = true;
            let p = spawn_point(self.queen.pos, self.bounds(), &mut self.rng, |p| {
                self.walls.is_wall(p)
            });
            // A flat health pool: the boss is the act's fixed exam, not
            // another notch on the Response level.
            let scale = self.cfg.boss_health;
            let ok = self.spawn(Enemy::new(EnemyKind::Cleaner, p, scale));
            log::info!(
                "the Cleaner arrives at {:.0}s (mass {}, spawned: {ok})",
                self.elapsed,
                self.colony.mass()
            );
            self.narrate("THE CLEANER IS HERE.");
        }
        // Regular threats, paused while the boss is on the board.
        if self.boss().is_none() && !self.boss_arrived {
            let alive = self.enemies.len();
            for kind in self.response.next_spawns(dt, alive) {
                let p = spawn_point(self.queen.pos, self.bounds(), &mut self.rng, |p| {
                    self.walls.is_wall(p)
                });
                let e = Enemy::new(kind, p, self.response.health_scale());
                self.spawn(e);
            }
        }

        let target = self.queen.pos;
        let bounds = self.bounds();
        let comet = form.filter(|f| f.kind == FormKind::Comet as u32);
        let center = self.game.swarm_center.unwrap_or(self.game.focal);
        let reach = self.game.swarm_radius + 2.0;
        let mut queen_dps = 0.0;
        for e in &mut self.enemies {
            e.update(dt, target, bounds);
            // A Comet ploughing into an enemy knocks it back.
            if let Some(c) = comet
                && e.pos.distance(center) < reach + e.body_radius()
            {
                e.knock_back(c.dir);
            }
            for z in e.zones().into_iter().flatten() {
                if z.is_active() && z.contains(self.queen.pos) {
                    queen_dps += z.queen_dps;
                }
            }
        }
        if queen_dps > 0.0 {
            let shield = if form.is_some_and(|f| f.kind == FormKind::Shield as u32) {
                SHIELD_DAMAGE_FACTOR
            } else {
                1.0
            };
            self.queen
                .damage(queen_dps * guard_factor(self.guards) * shield * dt);
        }

        // Clear out the beaten.
        let boss_beaten = self
            .enemies
            .iter()
            .any(|e| e.kind == EnemyKind::Cleaner && e.is_dead());
        self.enemies.retain(|e| !e.is_dead());
        if boss_beaten {
            log::info!("the Cleaner is beaten at {:.0}s", self.elapsed);
            self.enemies.clear();
            self.state = RunState::Transition { t: 0.0 };
            self.narrate("ACT 1 CLEARED. THE GARDEN AWAITS.");
        }
    }

    fn bounds(&self) -> (Vec2, Vec2) {
        (self.game.bounds.min, self.game.bounds.max)
    }

    /// Adds an enemy if there are free GPU slots for it.
    pub fn spawn(&mut self, mut e: Enemy) -> bool {
        let used: Vec<usize> = self
            .enemies
            .iter()
            .flat_map(|e| e.slots.into_iter().flatten())
            .collect();
        let free: Vec<usize> = (0..MAX_THREATS).filter(|s| !used.contains(s)).collect();
        if free.len() < e.slots_needed() {
            return false;
        }
        let needed = e.slots_needed();
        for (slot, &f) in e.slots.iter_mut().zip(&free).take(needed) {
            *slot = Some(f);
        }
        e.sync_hits(&self.hits);
        self.enemies.push(e);
        true
    }

    fn threat_uniforms(&self) -> [Threat; MAX_THREATS] {
        let mut out = [Threat::default(); MAX_THREATS];
        for e in &self.enemies {
            let u = e.threat_uniforms();
            for (k, slot) in e.slots.iter().enumerate() {
                if let Some(s) = *slot {
                    out[s] = u[k];
                }
            }
        }
        out
    }

    /// Aim direction for a Form: toward the pointer, else along the stick,
    /// else toward the focal point, else the way the Queen faces.
    fn aim(&self, controls: &Controls) -> Vec2 {
        let from = self.game.swarm_center.unwrap_or(self.game.focal);
        controls
            .aim
            .map(|p| p - from)
            .filter(|d| d.length_squared() > 1e-4)
            .or_else(|| Some(controls.move_dir).filter(|d| d.length_squared() > 1e-4))
            .or_else(|| Some(self.game.focal - from).filter(|d| d.length_squared() > 1.0))
            .unwrap_or(self.queen.heading)
            .normalize_or(Vec2::Y)
    }

    /// The active Form's shape for the GPU, if one is held.
    pub fn form_shape(&self) -> Option<FormShape> {
        let a = self.forms.active?;
        let center = self.game.swarm_center.unwrap_or(self.game.focal);
        let r = self.game.swarm_radius.max(2.0);
        let (origin, size) = match a.kind {
            FormKind::Spear => (center, (r * 3.0).max(8.0)),
            // A shell around the Queen big enough for the whole swarm.
            FormKind::Shield => (self.queen.pos, (QUEEN_RADIUS * 2.4).max(r * 0.7)),
            FormKind::Ring => (center, r * 1.3 + 3.0),
            FormKind::Comet | FormKind::Bloom => (center, r),
        };
        Some(FormShape {
            kind: a.kind as u32,
            origin,
            dir: a.dir,
            size,
            t: a.t(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::queen::MAX_HEALTH;
    use crate::threats::Phase;

    const DT: f32 = 1.0 / 60.0;

    fn run(cfg: RunConfig) -> Run {
        Run::new(&Level::kitchen(1), &cfg, 16.0 / 9.0)
    }

    fn calm() -> RunConfig {
        RunConfig {
            threats: false,
            ..Default::default()
        }
    }

    #[test]
    fn title_waits_for_start() {
        let mut r = run(RunConfig {
            title: true,
            ..Default::default()
        });
        r.update(&Controls::default(), DT);
        assert_eq!(r.state, RunState::Title);
        assert_eq!(r.elapsed, 0.0);
        r.update(
            &Controls {
                start: true,
                ..Default::default()
            },
            DT,
        );
        assert_eq!(r.state, RunState::Playing);
    }

    #[test]
    fn eating_feeds_hatching_at_the_queen() {
        let mut r = run(calm());
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
    fn forms_cost_mass_and_shape_the_frame() {
        let mut r = run(calm());
        r.colony.grant(90.0); // mass 100
        let press = Controls {
            form: [true, false],
            aim: Some(r.queen.pos + Vec2::new(10.0, 0.0)),
            ..Default::default()
        };
        let frame = r.update(&press, DT);
        let shape = frame.form.expect("spear is active");
        assert_eq!(shape.kind, FormKind::Spear as u32);
        assert!(shape.dir.x > 0.99);
        assert_eq!(
            r.colony.mass(),
            95,
            "a Spear costs 5% of 100, from the bank"
        );
        assert_eq!(frame.sacrifice, 0);
        assert_eq!(r.response.spear_uses, 1);
        // Holding the button does not fire it again.
        let frame = r.update(&press.held(), DT);
        assert!(frame.form.is_some());
        assert_eq!(r.colony.mass(), 95);
    }

    #[test]
    fn build_is_held_while_playing() {
        let mut r = run(calm());
        let frame = r.update(
            &Controls {
                build: true,
                ..Default::default()
            },
            DT,
        );
        assert!(frame.build);
    }

    /// A run with one spray parked on the Queen, mid-attack.
    fn sprayed(shield: bool, guards: u32) -> Run {
        let mut r = run(RunConfig {
            boss_mass: u32::MAX,
            ..Default::default()
        });
        r.colony.grant(200.0);
        r.guards = guards;
        if shield {
            r.update(
                &Controls {
                    form: [false, true],
                    ..Default::default()
                },
                DT,
            );
        }
        let mut e = Enemy::new(EnemyKind::Spray, r.queen.pos + Vec2::new(-5.0, 0.0), 1.0);
        e.aim = Vec2::X;
        e.phase = Phase::Attack(0.0);
        assert!(r.spawn(e));
        r
    }

    #[test]
    fn spray_hurts_the_queen_and_can_end_the_run() {
        let mut r = sprayed(false, 0);
        for _ in 0..(30.0 / DT) as usize {
            // Keep the spray attacking her.
            let q = r.queen.pos;
            for e in &mut r.enemies {
                e.pos = q + Vec2::new(-5.0, 0.0);
                e.aim = Vec2::X;
                e.phase = Phase::Attack(0.0);
            }
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
    fn shield_and_guards_protect_the_queen() {
        let mut exposed = sprayed(false, 0);
        let mut shielded = sprayed(true, 0);
        let mut guarded = sprayed(false, 60);
        for r in [&mut exposed, &mut shielded, &mut guarded] {
            for _ in 0..90 {
                r.update(&Controls::default(), DT);
            }
        }
        assert!(
            exposed.queen.health < MAX_HEALTH - 10.0,
            "{}",
            exposed.queen.health
        );
        assert!(shielded.queen.health > exposed.queen.health + 10.0);
        assert!(guarded.queen.health > exposed.queen.health + 10.0);
    }

    #[test]
    fn threats_arrive_as_the_swarm_grows() {
        let mut r = run(RunConfig {
            boss_mass: u32::MAX,
            ..Default::default()
        });
        for _ in 0..600 {
            r.update(&Controls::default(), DT);
        }
        assert!(
            r.enemies.is_empty(),
            "ten critters do not alarm the kitchen"
        );
        r.colony.grant(500.0);
        for _ in 0..600 {
            r.update(&Controls::default(), DT);
        }
        assert!(!r.enemies.is_empty());
        assert!(r.narration.is_some() || r.response.level > 0);
        // Every enemy has a GPU slot and appears in the frame.
        let frame = r.update(&Controls::default(), DT);
        let bodies = frame.threats.iter().filter(|t| t.body_radius > 0.0).count();
        assert_eq!(bodies, r.enemies.len());
    }

    #[test]
    fn beating_the_cleaner_clears_the_act() {
        let mut r = run(RunConfig {
            boss_mass: 0,
            ..Default::default()
        });
        r.update(&Controls::default(), DT);
        assert!(
            r.boss().is_some(),
            "the Cleaner comes at once with boss_mass 0"
        );
        let slot = r.boss().unwrap().slots[0].unwrap();
        // The swarm lands enough contact damage to finish it.
        let mut hits = [0u32; MAX_THREATS];
        hits[slot] = 10_000_000;
        r.observe(
            &SwarmSummary {
                alive: 10,
                enemy_hits: hits,
                ..Default::default()
            },
            2.5,
        );
        let frame = r.update(&Controls::default(), DT);
        assert!(matches!(r.state, RunState::Transition { .. }));
        assert!(r.enemies.is_empty());
        assert!(frame.merge, "the transition consolidates the swarm");
        assert!(!r.update(&Controls::default(), DT).merge, "once");
        for _ in 0..(TRANSITION_SECS / DT) as usize + 2 {
            r.update(&Controls::default(), DT);
        }
        assert_eq!(r.state, RunState::Won);
        assert!(r.is_over());
    }
}
