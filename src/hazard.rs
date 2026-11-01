//! The prototype hazard: a mop that patrols the kitchen floor. It wipes out
//! critters it touches and hurts the Queen (Step 4 replaces it with real
//! enemies and the Cleaner boss).

use glam::Vec2;

use crate::sim::HazardZone;

pub const MOP_RADIUS: f32 = 5.0;
pub const MOP_SPEED: f32 = 7.0;
/// Queen damage per second while she is under the mop, before guard reduction.
pub const MOP_QUEEN_DPS: f32 = 30.0;
/// Seconds the mop pauses at each stop on its route.
pub const MOP_PAUSE: f32 = 1.5;
/// Seconds of warning before the mop first appears.
pub const MOP_WARNING: f32 = 4.0;

#[derive(Clone, Debug)]
pub struct Mop {
    route: Vec<Vec2>,
    next: usize,
    pub pos: Vec2,
    /// Seconds until it appears; it is harmless until then.
    pub delay: f32,
    pause: f32,
}

impl Mop {
    pub fn new(route: Vec<Vec2>, delay: f32) -> Self {
        let pos = route.first().copied().unwrap_or(Vec2::ZERO);
        Self {
            next: 1.min(route.len().saturating_sub(1)),
            route,
            pos,
            delay,
            pause: 0.0,
        }
    }

    pub fn is_active(&self) -> bool {
        self.delay <= 0.0 && !self.route.is_empty()
    }

    /// 0..1 while the warning is showing (1 = about to appear), else 0.
    pub fn warning(&self) -> f32 {
        if self.delay > 0.0 && self.delay <= MOP_WARNING {
            1.0 - self.delay / MOP_WARNING
        } else {
            0.0
        }
    }

    pub fn update(&mut self, dt: f32) {
        if self.delay > 0.0 {
            self.delay -= dt;
            return;
        }
        if self.route.len() < 2 {
            return;
        }
        if self.pause > 0.0 {
            self.pause -= dt;
            return;
        }
        let target = self.route[self.next];
        let to = target - self.pos;
        let step = MOP_SPEED * dt;
        if to.length() <= step {
            self.pos = target;
            self.next = (self.next + 1) % self.route.len();
            self.pause = MOP_PAUSE;
        } else {
            self.pos += to.normalize() * step;
        }
    }

    pub fn zone(&self) -> Option<HazardZone> {
        self.is_active().then_some(HazardZone {
            center: self.pos,
            radius: MOP_RADIUS,
        })
    }

    /// Whether a circle of `radius` at `p` overlaps the active mop.
    pub fn touches(&self, p: Vec2, radius: f32) -> bool {
        self.is_active() && self.pos.distance(p) < MOP_RADIUS + radius
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn waits_warns_then_patrols_in_a_loop() {
        let route = vec![Vec2::ZERO, Vec2::new(14.0, 0.0)];
        let mut mop = Mop::new(route, MOP_WARNING + 1.0);
        assert!(!mop.is_active() && mop.warning() == 0.0);
        mop.update(2.0);
        assert!(mop.warning() > 0.0);
        mop.update(MOP_WARNING);
        assert!(mop.is_active());
        assert!(mop.zone().is_some());
        // 14 units at MOP_SPEED = 2 s to the far end, then a pause, then back.
        for _ in 0..(2.2 * 60.0) as usize {
            mop.update(1.0 / 60.0);
        }
        assert_eq!(mop.pos, Vec2::new(14.0, 0.0));
        for _ in 0..((MOP_PAUSE + 2.2) * 60.0) as usize {
            mop.update(1.0 / 60.0);
        }
        assert_eq!(mop.pos, Vec2::ZERO);
        assert!(mop.touches(Vec2::new(5.5, 0.0), 1.0));
        assert!(!mop.touches(Vec2::new(7.0, 0.0), 1.0));
    }
}
