//! The mass economy (§4.3): mass is the one number that is health, score and
//! (from Step 3) Form currency. Food eaten on the GPU becomes banked mass; banked
//! mass becomes hatch requests; hatchlings appear at the Queen.
//!
//! Everything here works from the cumulative counters in [`SwarmSummary`], so
//! late or skipped readbacks never lose or double-count anything.

use crate::sim::feedback::SwarmSummary;
use crate::sim::material::FULL;

/// Critters one full material cell of food is worth.
pub const CRITTERS_PER_CELL: f64 = 1.0;
/// Hatchlings placed per simulation step at most (a steady trickle).
pub const MAX_HATCH_PER_STEP: u32 = 16;

#[derive(Clone, Debug, Default)]
pub struct Colony {
    /// Food converted to mass but not yet hatched, in critters.
    pub bank: f64,
    /// Living critters, as of the latest summary.
    pub alive: u32,
    /// Critters lost so far (hazards, drowning).
    pub lost: u32,
    /// Critters spent on Forms so far.
    pub spent: u32,
    /// Food eaten so far, in full cells.
    pub food_eaten: f64,
    /// Highest mass reached.
    pub peak_mass: u32,
    eaten_seen: u32,
    requested_total: u64,
    confirmed_total: u32,
    sacrifice_requested: u64,
    sacrifice_confirmed: u32,
}

impl Colony {
    pub fn new(initial_critters: u32) -> Self {
        Self {
            alive: initial_critters,
            peak_mass: initial_critters,
            ..Self::default()
        }
    }

    /// Hatch requests sent but not yet confirmed by the GPU.
    pub fn in_flight(&self) -> u32 {
        self.requested_total
            .saturating_sub(self.confirmed_total as u64) as u32
    }

    /// Critters asked to be spent but not yet confirmed gone.
    pub fn sacrifice_pending(&self) -> u32 {
        self.sacrifice_requested
            .saturating_sub(self.sacrifice_confirmed as u64) as u32
    }

    /// Total mass: living critters, hatchlings on the way, and the bank, less
    /// critters already spent but still on the GPU.
    pub fn mass(&self) -> u32 {
        (self.alive + self.in_flight() + self.bank.floor() as u32)
            .saturating_sub(self.sacrifice_pending())
    }

    /// Pays `cost` mass: from the bank first, then by giving up critters.
    /// Returns how many critters must be sacrificed.
    pub fn spend(&mut self, cost: u32) -> u32 {
        let from_bank = (self.bank.floor() as u32).min(cost);
        self.bank -= from_bank as f64;
        let rest = cost - from_bank;
        self.sacrifice_requested += rest as u64;
        rest
    }

    /// Folds in a summary from the GPU.
    pub fn observe(&mut self, s: &SwarmSummary) {
        let new_food = s.eaten_total.wrapping_sub(self.eaten_seen) as f64 / FULL as f64;
        self.eaten_seen = s.eaten_total;
        self.food_eaten += new_food;
        self.bank += new_food * CRITTERS_PER_CELL;
        self.alive = s.alive;
        self.lost = s.died_total;
        self.spent = s.sacrificed_total;
        self.sacrifice_confirmed = s.sacrificed_total;
        self.confirmed_total = s.hatched_total;
        self.peak_mass = self.peak_mass.max(self.mass());
    }

    /// Takes hatch requests out of the bank, never more than will fit.
    pub fn take_hatch(&mut self, capacity: u32) -> u32 {
        let free = capacity.saturating_sub(self.alive + self.in_flight());
        let n = (self.bank.floor() as u32).min(free).min(MAX_HATCH_PER_STEP);
        self.bank -= n as f64;
        self.requested_total += n as u64;
        n
    }

    /// Adds mass directly (developer cheat).
    pub fn grant(&mut self, critters: f64) {
        self.bank += critters;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn summary(alive: u32, hatched: u32, eaten_cells: u32, died: u32) -> SwarmSummary {
        SwarmSummary {
            alive,
            hatched_total: hatched,
            eaten_total: eaten_cells * FULL,
            died_total: died,
            ..Default::default()
        }
    }

    #[test]
    fn food_becomes_hatchlings_without_double_counting() {
        let mut c = Colony::new(10);
        c.observe(&summary(10, 0, 20, 0)); // 20 cells -> 20 critters banked
        assert_eq!(c.mass(), 30);
        let first = c.take_hatch(1_000);
        assert_eq!(first, MAX_HATCH_PER_STEP);
        assert_eq!(c.mass(), 30, "requests in flight still count");
        // A stale summary (before the hatch landed) changes nothing.
        c.observe(&summary(10, 0, 20, 0));
        assert_eq!(c.mass(), 30);
        // The hatch lands.
        c.observe(&summary(10 + first, first, 20, 0));
        assert_eq!(c.in_flight(), 0);
        assert_eq!(c.mass(), 30);
        // The rest of the bank follows.
        let second = c.take_hatch(1_000);
        assert_eq!(first + second, 20);
    }

    #[test]
    fn spending_uses_the_bank_then_critters() {
        let mut c = Colony::new(100);
        c.grant(5.0);
        assert_eq!(c.mass(), 105);
        assert_eq!(c.spend(12), 7, "5 from the bank, 7 critters");
        assert_eq!(c.mass(), 93, "pending sacrifices already count");
        c.observe(&SwarmSummary {
            alive: 93,
            sacrificed_total: 7,
            ..Default::default()
        });
        assert_eq!((c.mass(), c.spent, c.sacrifice_pending()), (93, 7, 0));
    }

    #[test]
    fn deaths_reduce_mass_and_capacity_limits_hatching() {
        let mut c = Colony::new(100);
        c.observe(&summary(90, 0, 0, 10));
        assert_eq!((c.mass(), c.lost), (90, 10));
        c.grant(50.0);
        assert_eq!(c.take_hatch(95), 5, "only 5 free slots");
        assert_eq!(c.take_hatch(95), 0);
        assert_eq!(c.peak_mass, 100);
    }
}
