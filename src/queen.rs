//! The Queen: the run's one weak point. New critters hatch around her, so the
//! swarm grows from its middle; if she dies, the run is over (§4.3).

use glam::Vec2;

/// Radius of her body in world units (a critter is ~0.45 long).
pub const QUEEN_RADIUS: f32 = 1.3;
pub const MAX_HEALTH: f32 = 100.0;
/// Seconds she clings on at zero health before recovering.
pub const LAST_STAND_SECS: f32 = 3.0;
/// Damage she can take during a last stand before she falls.
pub const LAST_STAND_TOLERANCE: f32 = 10.0;
/// Health she recovers to after surviving a last stand.
pub const LAST_STAND_RECOVERY: f32 = 25.0;
/// Her top speed as a fraction of the swarm's: she lags behind the action.
pub const SPEED_FACTOR: f32 = 0.6;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum QueenState {
    Healthy,
    /// At zero health; survives if she takes little more damage in time.
    LastStand {
        remaining: f32,
        taken: f32,
    },
    Dead,
}

#[derive(Clone, Debug)]
pub struct Queen {
    pub pos: Vec2,
    /// Unit facing direction.
    pub heading: Vec2,
    pub health: f32,
    pub state: QueenState,
    /// 1 right after taking damage, fading to 0 (for rendering).
    pub hurt_flash: f32,
}

impl Queen {
    pub fn new(pos: Vec2) -> Self {
        Self {
            pos,
            heading: Vec2::Y,
            health: MAX_HEALTH,
            state: QueenState::Healthy,
            hurt_flash: 0.0,
        }
    }

    pub fn is_dead(&self) -> bool {
        self.state == QueenState::Dead
    }

    /// Follows `target` (the swarm's centre) at up to `max_speed`, sliding
    /// along walls. Also ticks the last-stand timer.
    pub fn update(
        &mut self,
        target: Option<Vec2>,
        max_speed: f32,
        dt: f32,
        is_wall: impl Fn(Vec2) -> bool,
    ) {
        self.hurt_flash = (self.hurt_flash - dt * 3.0).max(0.0);
        if let QueenState::LastStand { remaining, taken } = self.state {
            let remaining = remaining - dt;
            self.state = if remaining <= 0.0 {
                self.health = LAST_STAND_RECOVERY;
                QueenState::Healthy
            } else {
                QueenState::LastStand { remaining, taken }
            };
        }
        if self.is_dead() {
            return;
        }
        let Some(target) = target else { return };
        // Ease toward the target: quick when far, settling when close.
        let vel = ((target - self.pos) * 1.5).clamp_length_max(max_speed);
        if vel.length_squared() > 1e-4 {
            self.heading = self
                .heading
                .lerp(vel.normalize(), (dt * 6.0).min(1.0))
                .normalize_or(Vec2::Y);
        }
        let next = self.pos + vel * dt;
        if !is_wall(next) {
            self.pos = next;
        } else if !is_wall(Vec2::new(next.x, self.pos.y)) {
            self.pos.x = next.x;
        } else if !is_wall(Vec2::new(self.pos.x, next.y)) {
            self.pos.y = next.y;
        }
    }

    pub fn damage(&mut self, amount: f32) {
        if amount <= 0.0 {
            return;
        }
        self.hurt_flash = 1.0;
        match self.state {
            QueenState::Healthy => {
                self.health -= amount;
                if self.health <= 0.0 {
                    self.health = 0.0;
                    self.state = QueenState::LastStand {
                        remaining: LAST_STAND_SECS,
                        taken: 0.0,
                    };
                }
            }
            QueenState::LastStand { remaining, taken } => {
                let taken = taken + amount;
                self.state = if taken > LAST_STAND_TOLERANCE {
                    QueenState::Dead
                } else {
                    QueenState::LastStand { remaining, taken }
                };
            }
            QueenState::Dead => {}
        }
    }
}

/// How much of incoming damage reaches the Queen given her guard: critters
/// close around her absorb most of it, so formation is defence.
pub fn guard_factor(guards: u32) -> f32 {
    1.0 / (1.0 + guards as f32 / 20.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f32 = 1.0 / 60.0;

    #[test]
    fn follows_target_and_respects_walls() {
        let mut q = Queen::new(Vec2::ZERO);
        for _ in 0..600 {
            q.update(Some(Vec2::new(10.0, 0.0)), 5.0, DT, |p| p.x > 6.0);
        }
        assert!((q.pos.x - 6.0).abs() < 0.2, "{}", q.pos);
        assert!(q.heading.x > 0.9);
    }

    #[test]
    fn last_stand_then_recovery() {
        let mut q = Queen::new(Vec2::ZERO);
        q.damage(150.0);
        assert!(matches!(q.state, QueenState::LastStand { .. }));
        q.damage(5.0); // within tolerance
        for _ in 0..(LAST_STAND_SECS / DT) as usize + 2 {
            q.update(None, 1.0, DT, |_| false);
        }
        assert_eq!(q.state, QueenState::Healthy);
        assert_eq!(q.health, LAST_STAND_RECOVERY);
    }

    #[test]
    fn dies_if_hit_hard_during_last_stand() {
        let mut q = Queen::new(Vec2::ZERO);
        q.damage(MAX_HEALTH);
        q.damage(LAST_STAND_TOLERANCE + 1.0);
        assert!(q.is_dead());
        q.update(Some(Vec2::new(5.0, 0.0)), 5.0, DT, |_| false);
        assert_eq!(q.pos, Vec2::ZERO, "the dead do not walk");
    }

    #[test]
    fn guards_absorb_damage() {
        assert_eq!(guard_factor(0), 1.0);
        assert!(guard_factor(20) == 0.5);
        assert!(guard_factor(200) < 0.1);
    }
}
