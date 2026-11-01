//! Threats (§4.3–4.4): the kitchen fights back.
//!
//! Enemies are CPU entities. Each one occupies a GPU [`Threat`] slot: its body
//! takes contact damage from critters (reported back as cumulative counters),
//! and its danger zone (a cone of spray, a wiping lane) slows and kills
//! critters and hurts the Queen. Every attack is telegraphed first.
//!
//! The [`Response`] level paces spawns with the swarm's growth and counters
//! the player's habits; the [`EnemyKind::Cleaner`] boss closes the act.

use glam::Vec2;

use crate::sim::{MAX_THREATS, Threat, zone};
use crate::util::SplitMix64;

/// Appearance codes for the renderer.
pub mod look {
    pub const NONE: u32 = 0;
    pub const SPRAY: u32 = 1;
    pub const CLOTH: u32 = 2;
    pub const CLEANER: u32 = 3;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnemyKind {
    /// Telegraphs a cone, then mists it: kills some critters, slows the rest.
    Spray,
    /// Telegraphs a lane through the swarm, then wipes along it.
    Cloth,
    /// The act's boss: sprays, wipes, then double-sprays as it weakens.
    Cleaner,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Phase {
    Approach,
    Telegraph(f32),
    Attack(f32),
    Rest(f32),
}

/// What an enemy's zone looks like this step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Zone {
    pub kind: u32,
    pub origin: Vec2,
    pub dir: Vec2,
    pub len: f32,
    pub width: f32,
    pub lethality: f32,
    pub slow: f32,
    /// 0 off, 1 telegraphing, 2 active.
    pub on: u32,
    pub telegraph: f32,
    /// Queen damage per second while she is inside the active zone.
    pub queen_dps: f32,
}

impl Zone {
    pub fn contains(&self, p: Vec2) -> bool {
        let d = p - self.origin;
        match self.kind {
            zone::CIRCLE => d.length() < self.width,
            zone::CONE => {
                let len = d.length();
                len < self.len && len > 1e-3 && d.dot(self.dir) / len > self.width.cos()
            }
            zone::LANE => {
                let h = d.dot(self.dir).clamp(0.0, self.len);
                (d - self.dir * h).length() < self.width
            }
            _ => false,
        }
    }

    pub fn is_active(&self) -> bool {
        self.on == 2
    }
}

#[derive(Clone, Debug)]
pub struct Enemy {
    pub kind: EnemyKind,
    pub pos: Vec2,
    pub aim: Vec2,
    pub health: f32,
    pub max_health: f32,
    pub phase: Phase,
    /// GPU slots: the body's, and for the Cleaner a second for its extra zone.
    pub slots: [Option<usize>; 2],
    /// Lane for a wipe: start and end.
    lane: (Vec2, Vec2),
    /// Attacks so far (the Cleaner alternates patterns).
    attacks: u32,
    /// Knockback velocity (from a Comet), decaying.
    pub knock: Vec2,
    /// Seconds on the board. The Cleaner enrages as it grows.
    pub age: f32,
    hits_seen: [u32; 2],
}

pub const SPRAY_HEALTH: f32 = 120.0;
pub const CLOTH_HEALTH: f32 = 180.0;
pub const CLEANER_HEALTH: f32 = 2000.0;

impl Enemy {
    pub fn new(kind: EnemyKind, pos: Vec2, health_scale: f32) -> Self {
        let base = match kind {
            EnemyKind::Spray => SPRAY_HEALTH,
            EnemyKind::Cloth => CLOTH_HEALTH,
            EnemyKind::Cleaner => CLEANER_HEALTH,
        };
        let health = base * health_scale;
        Self {
            kind,
            pos,
            aim: Vec2::X,
            health,
            max_health: health,
            phase: Phase::Approach,
            slots: [None; 2],
            lane: (pos, pos),
            attacks: 0,
            knock: Vec2::ZERO,
            age: 0.0,
            hits_seen: [0; 2],
        }
    }

    pub fn body_radius(&self) -> f32 {
        match self.kind {
            EnemyKind::Spray => 2.0,
            EnemyKind::Cloth => 2.6,
            EnemyKind::Cleaner => 4.5,
        }
    }

    pub fn is_dead(&self) -> bool {
        self.health <= 0.0
    }

    pub fn health_frac(&self) -> f32 {
        (self.health / self.max_health).clamp(0.0, 1.0)
    }

    /// Slots this enemy needs.
    pub fn slots_needed(&self) -> usize {
        if self.kind == EnemyKind::Cleaner {
            2
        } else {
            1
        }
    }

    /// Cleaner phase by health: 1 sprays, 2 alternates spray and wipe,
    /// 3 double-sprays faster.
    pub fn boss_phase(&self) -> u32 {
        match self.health_frac() {
            f if f > 0.6 => 1,
            f if f > 0.25 => 2,
            _ => 3,
        }
    }

    /// Applies contact damage from cumulative GPU counters.
    pub fn observe_hits(&mut self, hits: &[u32; MAX_THREATS]) {
        for (k, slot) in self.slots.iter().enumerate() {
            if let Some(s) = *slot {
                let delta = hits[s].wrapping_sub(self.hits_seen[k]);
                self.hits_seen[k] = hits[s];
                self.health -= delta as f32 / crate::sim::HIT_SCALE;
            }
        }
    }

    /// Records the counters' current values without applying them (on slot
    /// assignment, so an earlier occupant's hits don't count).
    pub fn sync_hits(&mut self, hits: &[u32; MAX_THREATS]) {
        for (k, slot) in self.slots.iter().enumerate() {
            if let Some(s) = *slot {
                self.hits_seen[k] = hits[s];
            }
        }
    }

    fn timings(&self) -> (f32, f32, f32) {
        // (telegraph, attack, rest) seconds
        match self.kind {
            EnemyKind::Spray => (1.2, 1.6, 2.0),
            EnemyKind::Cloth => (1.5, 1.2, 2.5),
            EnemyKind::Cleaner => match self.boss_phase() {
                3 => (0.9, 1.6, 1.2),
                _ => (1.3, 1.8, 1.8),
            },
        }
    }

    fn is_wiping(&self) -> bool {
        match self.kind {
            EnemyKind::Cloth => true,
            EnemyKind::Cleaner => self.boss_phase() == 2 && self.attacks % 2 == 1,
            EnemyKind::Spray => false,
        }
    }

    /// Advances behaviour. `target` is what it attacks (the Queen).
    /// The Cleaner's enrage: ignoring it gets more dangerous every second.
    pub fn enrage(&self) -> f32 {
        if self.kind == EnemyKind::Cleaner {
            1.0 + self.age / 30.0
        } else {
            1.0
        }
    }

    pub fn update(&mut self, dt: f32, target: Vec2, bounds: (Vec2, Vec2)) {
        self.age += dt;
        if self.knock.length_squared() > 0.01 {
            self.pos += self.knock * dt;
            self.knock *= (-dt * 4.0).exp();
        }
        let (tele, attack, rest) = self.timings();
        let speed = match self.kind {
            EnemyKind::Spray => 7.0,
            EnemyKind::Cloth => 8.0,
            EnemyKind::Cleaner => (3.5 + self.age * 0.05).min(9.0),
        };
        let standoff = match self.kind {
            EnemyKind::Spray => 12.0,
            EnemyKind::Cloth => 0.0,
            EnemyKind::Cleaner => 11.0,
        };
        self.phase = match self.phase {
            Phase::Approach => {
                let goal = if self.is_wiping() {
                    // Line up at one side of the target for a sweep across it,
                    // from a different side each time.
                    let across = [Vec2::X, Vec2::Y, -Vec2::X, -Vec2::Y][self.attacks as usize % 4];
                    target + across * 14.0
                } else {
                    target + (self.pos - target).normalize_or(Vec2::X) * standoff
                };
                let to = goal - self.pos;
                self.pos += to.clamp_length_max(speed * dt);
                // Sprayers fire as soon as the target is in reach, so a moving
                // target cannot dodge forever by never letting them settle.
                let in_reach = !self.is_wiping() && self.pos.distance(target) < standoff + 2.0;
                if to.length() < 1.5 || in_reach {
                    self.aim = (target - self.pos).normalize_or(Vec2::X);
                    if self.is_wiping() {
                        self.lane = (self.pos, target + (target - self.pos));
                        self.aim = (self.lane.1 - self.lane.0).normalize_or(Vec2::X);
                    }
                    Phase::Telegraph(0.0)
                } else {
                    Phase::Approach
                }
            }
            Phase::Telegraph(t) if t + dt >= tele => Phase::Attack(0.0),
            Phase::Telegraph(t) => {
                if !self.is_wiping() {
                    // Track the target a little while winding up.
                    let want = (target - self.pos).normalize_or(self.aim);
                    self.aim = self.aim.lerp(want, (dt * 1.5).min(1.0)).normalize_or(want);
                }
                Phase::Telegraph(t + dt)
            }
            Phase::Attack(t) => {
                if self.is_wiping() {
                    let (a, b) = self.lane;
                    let along = ((t + dt) / attack).min(1.0);
                    self.pos = a.lerp(b, along);
                }
                if t + dt >= attack {
                    self.attacks += 1;
                    Phase::Rest(0.0)
                } else {
                    Phase::Attack(t + dt)
                }
            }
            Phase::Rest(t) if t + dt >= rest => Phase::Approach,
            Phase::Rest(t) => Phase::Rest(t + dt),
        };
        self.pos = self.pos.clamp(bounds.0, bounds.1);
    }

    /// The danger zones this enemy projects now (up to two).
    pub fn zones(&self) -> [Option<Zone>; 2] {
        let (tele, _, _) = self.timings();
        let (on, telegraph) = match self.phase {
            Phase::Telegraph(t) => (1, (t / tele).clamp(0.0, 1.0)),
            Phase::Attack(_) => (2, 1.0),
            _ => return [None, None],
        };
        let enrage = self.enrage();
        let cone = |dir: Vec2, len: f32, half: f32, lethality: f32, dps: f32| Zone {
            kind: zone::CONE,
            origin: self.pos,
            dir,
            len,
            width: half,
            lethality,
            slow: 0.9,
            on,
            telegraph,
            queen_dps: dps * enrage,
        };
        if self.is_wiping() {
            let (a, b) = self.lane;
            let z = if on == 2 {
                // The cloth itself, sweeping along the lane.
                Zone {
                    kind: zone::CIRCLE,
                    origin: self.pos,
                    dir: self.aim,
                    len: 0.0,
                    width: 3.4,
                    lethality: 0.08,
                    slow: 0.6,
                    on,
                    telegraph,
                    queen_dps: 35.0,
                }
            } else {
                Zone {
                    kind: zone::LANE,
                    origin: a,
                    dir: self.aim,
                    len: a.distance(b),
                    width: 3.4,
                    lethality: 0.0,
                    slow: 1.0,
                    on,
                    telegraph,
                    queen_dps: 0.0,
                }
            };
            return [Some(z), None];
        }
        match self.kind {
            EnemyKind::Cleaner if self.boss_phase() == 3 => {
                let rot = |a: f32| Vec2::from_angle(a).rotate(self.aim);
                [
                    Some(cone(rot(0.45), 20.0, 0.45, 0.004, 18.0)),
                    Some(cone(rot(-0.45), 20.0, 0.45, 0.004, 18.0)),
                ]
            }
            EnemyKind::Cleaner => [Some(cone(self.aim, 20.0, 0.6, 0.004, 20.0)), None],
            _ => [Some(cone(self.aim, 15.0, 0.45, 0.003, 14.0)), None],
        }
    }

    /// GPU slot contents: the body (with its first zone) and, for the
    /// Cleaner, a second zone without a body.
    pub fn threat_uniforms(&self) -> [Threat; 2] {
        let zones = self.zones();
        let look = match self.kind {
            EnemyKind::Spray => look::SPRAY,
            EnemyKind::Cloth => look::CLOTH,
            EnemyKind::Cleaner => look::CLEANER,
        };
        let mk = |z: Option<Zone>, body: f32, look: u32| {
            let mut t = Threat {
                pos: self.pos.to_array(),
                dir: self.aim.to_array(),
                body_radius: body,
                look,
                health: self.health_frac(),
                slow: 1.0,
                ..Default::default()
            };
            if let Some(z) = z {
                // Zones start at the body (the lane starts where the cloth is).
                t.pos = z.origin.to_array();
                t.zone_kind = z.kind;
                t.dir = z.dir.to_array();
                t.zone_len = z.len;
                t.zone_width = z.width;
                t.lethality = z.lethality;
                t.slow = z.slow;
                t.zone_on = z.on;
                t.telegraph = z.telegraph;
            }
            t
        };
        [
            mk(zones[0], self.body_radius(), look),
            mk(zones[1], 0.0, look::NONE),
        ]
    }

    /// Knocks the enemy back (Comet) and interrupts its attack.
    pub fn knock_back(&mut self, dir: Vec2) {
        self.knock = dir.normalize_or(Vec2::X) * 30.0;
        self.phase = Phase::Rest(0.0);
    }
}

/// Pressure that rises with the swarm, spawns threats and counters habits.
#[derive(Clone, Debug, Default)]
pub struct Response {
    pub level: u32,
    spawn_timer: f32,
    spawned: u32,
    pub shield_uses: u32,
    pub spear_uses: u32,
}

impl Response {
    /// Level for a swarm of `mass` after `elapsed` seconds: 0 below 100 mass,
    /// then +1 per doubling, plus one per two minutes survived.
    pub fn level_for(mass: u32, elapsed: f32) -> u32 {
        if mass < 100 {
            return 0;
        }
        let growth = (mass as f32 / 100.0).log2().floor() as u32 + 1;
        (growth + (elapsed / 120.0) as u32).min(10)
    }

    /// Enemies allowed alive at once.
    pub fn max_alive(&self) -> usize {
        if self.level == 0 {
            0
        } else {
            (1 + self.level as usize / 3).min(3)
        }
    }

    pub fn health_scale(&self) -> f32 {
        1.0 + 0.3 * self.level.saturating_sub(1) as f32
    }

    /// Updates the level; returns a narrator line if it changed.
    pub fn update_level(&mut self, mass: u32, elapsed: f32) -> Option<&'static str> {
        let level = Self::level_for(mass, elapsed);
        if level <= self.level {
            return None;
        }
        self.level = level;
        Some(match level {
            1 => "THE KITCHEN HAS NOTICED YOU.",
            2 => "SOMETHING REACHES FOR THE SPRAY.",
            3 => "THE KITCHEN IS GETTING SERIOUS.",
            4 => "IT KNOWS YOUR SHAPE NOW.",
            _ => "THE WHOLE HOUSE IS AWAKE.",
        })
    }

    /// Picks what to spawn next, if it is time and there is room. Counters
    /// habits: lots of Shield brings sprays in pairs; lots of Spear brings
    /// cloths, which sweep a line.
    pub fn next_spawns(&mut self, dt: f32, alive: usize) -> Vec<EnemyKind> {
        if self.level == 0 || alive >= self.max_alive() {
            return Vec::new();
        }
        self.spawn_timer -= dt;
        if self.spawn_timer > 0.0 {
            return Vec::new();
        }
        self.spawn_timer = 14.0 / (1.0 + 0.2 * self.level as f32);
        self.spawned += 1;
        let room = self.max_alive() - alive;
        if self.shield_uses > self.spear_uses + 1 && room >= 2 {
            vec![EnemyKind::Spray, EnemyKind::Spray]
        } else if self.spear_uses > self.shield_uses + 1 || self.spawned.is_multiple_of(3) {
            vec![EnemyKind::Cloth]
        } else {
            vec![EnemyKind::Spray]
        }
    }
}

/// A spawn point around `center`, inside `bounds`, not in a wall.
pub fn spawn_point(
    center: Vec2,
    bounds: (Vec2, Vec2),
    rng: &mut SplitMix64,
    is_wall: impl Fn(Vec2) -> bool,
) -> Vec2 {
    let inner = (bounds.0 + Vec2::splat(6.0), bounds.1 - Vec2::splat(6.0));
    for _ in 0..16 {
        let ang = rng.next_f32() * std::f32::consts::TAU;
        let p = (center + Vec2::from_angle(ang) * 32.0).clamp(inner.0, inner.1);
        if !is_wall(p) {
            return p;
        }
    }
    center
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f32 = 1.0 / 60.0;
    const BOUNDS: (Vec2, Vec2) = (Vec2::splat(-100.0), Vec2::splat(100.0));

    #[test]
    fn spray_approaches_telegraphs_then_attacks() {
        let mut e = Enemy::new(EnemyKind::Spray, Vec2::new(40.0, 0.0), 1.0);
        let target = Vec2::ZERO;
        let mut saw = (false, false);
        for _ in 0..(12.0 / DT) as usize {
            e.update(DT, target, BOUNDS);
            let z = e.zones()[0];
            match e.phase {
                Phase::Telegraph(_) => {
                    saw.0 = true;
                    assert_eq!(z.unwrap().on, 1, "telegraphs are harmless");
                }
                Phase::Attack(_) => {
                    assert!(saw.0, "never attacks without a telegraph first");
                    saw.1 = true;
                    let z = z.unwrap();
                    assert!(
                        z.is_active() && z.contains(target),
                        "the cone covers the target"
                    );
                }
                _ => {}
            }
        }
        assert!(saw.0 && saw.1);
    }

    #[test]
    fn cloth_wipes_across_the_target() {
        let mut e = Enemy::new(EnemyKind::Cloth, Vec2::new(0.0, 30.0), 1.0);
        let mut crossed = false;
        for _ in 0..(15.0 / DT) as usize {
            e.update(DT, Vec2::ZERO, BOUNDS);
            if let Phase::Telegraph(_) = e.phase {
                assert!(
                    e.zones()[0].unwrap().contains(Vec2::ZERO),
                    "the lane runs through the target"
                );
            }
            if let Phase::Attack(_) = e.phase {
                crossed |= e.pos.length() < 3.0;
            }
        }
        assert!(crossed, "the cloth sweeps over the target");
    }

    #[test]
    fn cleaner_changes_pattern_as_it_weakens() {
        let mut e = Enemy::new(EnemyKind::Cleaner, Vec2::ZERO, 1.0);
        assert_eq!(e.boss_phase(), 1);
        e.health = e.max_health * 0.5;
        assert_eq!(e.boss_phase(), 2);
        e.health = e.max_health * 0.1;
        assert_eq!(e.boss_phase(), 3);
        e.phase = Phase::Attack(0.0);
        let z = e.zones();
        assert!(z[0].is_some() && z[1].is_some(), "phase 3 double-sprays");
    }

    #[test]
    fn the_cleaner_enrages() {
        let mut e = Enemy::new(EnemyKind::Cleaner, Vec2::new(30.0, 0.0), 1.0);
        e.phase = Phase::Attack(0.0);
        let fresh = e.zones()[0].unwrap().queen_dps;
        e.age = 60.0;
        assert!((e.zones()[0].unwrap().queen_dps - fresh * 3.0).abs() < 1e-3);
    }

    #[test]
    fn contact_damage_comes_from_cumulative_counters() {
        let mut e = Enemy::new(EnemyKind::Spray, Vec2::ZERO, 1.0);
        e.slots[0] = Some(3);
        let mut hits = [0u32; MAX_THREATS];
        hits[3] = 50_000; // an earlier occupant's hits
        e.sync_hits(&hits);
        hits[3] = 150_000;
        e.observe_hits(&hits);
        assert!((e.health - (SPRAY_HEALTH - 100.0)).abs() < 1e-3);
    }

    #[test]
    fn zone_geometry() {
        let cone = Zone {
            kind: zone::CONE,
            origin: Vec2::ZERO,
            dir: Vec2::X,
            len: 10.0,
            width: 0.5,
            lethality: 0.0,
            slow: 1.0,
            on: 2,
            telegraph: 1.0,
            queen_dps: 0.0,
        };
        assert!(cone.contains(Vec2::new(5.0, 1.0)));
        assert!(!cone.contains(Vec2::new(5.0, 4.0)));
        assert!(!cone.contains(Vec2::new(-5.0, 0.0)));
        let lane = Zone {
            kind: zone::LANE,
            width: 2.0,
            ..cone
        };
        assert!(lane.contains(Vec2::new(9.0, 1.5)));
        assert!(lane.contains(Vec2::new(11.0, 0.0)), "a capsule: round ends");
        assert!(!lane.contains(Vec2::new(13.0, 0.0)));
    }

    #[test]
    fn response_escalates_with_mass_and_time() {
        assert_eq!(Response::level_for(60, 0.0), 0);
        assert_eq!(Response::level_for(100, 0.0), 1);
        assert_eq!(Response::level_for(400, 0.0), 3);
        assert_eq!(Response::level_for(400, 240.0), 5);
        let mut r = Response::default();
        assert!(r.update_level(500, 0.0).is_some());
        assert!(r.update_level(500, 0.0).is_none(), "announces once");
        assert!(r.max_alive() >= 1);
        assert!(!r.next_spawns(0.0, 0).is_empty());
        assert!(r.next_spawns(0.0, 0).is_empty(), "then waits");
    }

    #[test]
    fn response_counters_the_shield() {
        let mut r = Response {
            level: 6,
            shield_uses: 4,
            ..Default::default()
        };
        assert_eq!(
            r.next_spawns(1.0, 0),
            vec![EnemyKind::Spray, EnemyKind::Spray]
        );
    }
}
