//! Forms (§4.3): hold a shape and the swarm becomes a weapon or a tool. Each
//! Form costs a share of the swarm's mass and then cools down. Two slots are
//! equipped at a time.

use glam::Vec2;

/// Discriminants must match the `FORM_*` constants in `common.wgsl`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum FormKind {
    /// A line that pierces: critters in it eat solids 4x as fast.
    Spear = 1,
    /// A shell around the Queen: critters in it ignore hazards, and the
    /// Queen takes a fraction of the damage.
    Shield = 2,
    /// A closing circle: critters in it eat 3x as fast.
    Ring = 3,
    /// A dash: the swarm balls up and rushes along the aim.
    Comet = 4,
    /// A burst outward that covers ground; critters eat 2x as fast.
    Bloom = 5,
}

pub const ALL: [FormKind; 5] = [
    FormKind::Spear,
    FormKind::Shield,
    FormKind::Ring,
    FormKind::Comet,
    FormKind::Bloom,
];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FormDef {
    pub name: &'static str,
    /// Share of current mass spent on activation.
    pub cost_share: f32,
    /// Minimum cost in critters.
    pub min_cost: u32,
    /// Seconds the Form holds.
    pub duration: f32,
    /// Seconds after it ends before it can be used again.
    pub cooldown: f32,
}

impl FormKind {
    pub fn def(self) -> FormDef {
        let (name, cost_share, min_cost, duration, cooldown) = match self {
            FormKind::Spear => ("SPEAR", 0.04, 5, 2.5, 6.0),
            FormKind::Shield => ("SHIELD", 0.06, 8, 4.0, 10.0),
            FormKind::Ring => ("RING", 0.05, 6, 3.0, 8.0),
            FormKind::Comet => ("COMET", 0.03, 3, 0.8, 4.0),
            FormKind::Bloom => ("BLOOM", 0.08, 10, 1.0, 12.0),
        };
        FormDef {
            name,
            cost_share,
            min_cost,
            duration,
            cooldown,
        }
    }

    /// Mass this Form costs for a swarm of `mass`.
    pub fn cost(self, mass: u32) -> u32 {
        let d = self.def();
        ((mass as f32 * d.cost_share).ceil() as u32).max(d.min_cost)
    }

    /// The next Form in [`ALL`] (for cycling a slot).
    pub fn next(self) -> FormKind {
        let i = ALL.iter().position(|&k| k == self).unwrap_or(0);
        ALL[(i + 1) % ALL.len()]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActiveForm {
    pub kind: FormKind,
    pub slot: usize,
    pub elapsed: f32,
    /// Unit aim direction, fixed when the Form starts.
    pub dir: Vec2,
}

impl ActiveForm {
    /// Progress 0..1.
    pub fn t(&self) -> f32 {
        (self.elapsed / self.kind.def().duration).clamp(0.0, 1.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FormError {
    /// Another Form is being held.
    Busy,
    /// This slot is cooling down.
    Cooling,
    /// The swarm is too small to pay for it (it would spend everything).
    TooSmall,
}

#[derive(Clone, Debug)]
pub struct Forms {
    pub slots: [FormKind; 2],
    /// Seconds left before each slot is ready.
    pub cooldown: [f32; 2],
    pub active: Option<ActiveForm>,
}

impl Default for Forms {
    fn default() -> Self {
        // The prototype loadout from §10: Spear and Shield.
        Self {
            slots: [FormKind::Spear, FormKind::Shield],
            cooldown: [0.0; 2],
            active: None,
        }
    }
}

impl Forms {
    /// Starts the Form in `slot` aimed along `dir`. Returns its mass cost.
    pub fn try_activate(&mut self, slot: usize, mass: u32, dir: Vec2) -> Result<u32, FormError> {
        if self.active.is_some() {
            return Err(FormError::Busy);
        }
        if self.cooldown[slot] > 0.0 {
            return Err(FormError::Cooling);
        }
        let kind = self.slots[slot];
        let cost = kind.cost(mass);
        if cost >= mass {
            return Err(FormError::TooSmall);
        }
        self.active = Some(ActiveForm {
            kind,
            slot,
            elapsed: 0.0,
            dir: dir.normalize_or(Vec2::Y),
        });
        Ok(cost)
    }

    /// Switches a slot to the next Form (only while it is idle).
    pub fn cycle(&mut self, slot: usize) {
        if self.active.is_none_or(|a| a.slot != slot) {
            self.slots[slot] = self.slots[slot].next();
            if self.slots[0] == self.slots[1] {
                self.slots[slot] = self.slots[slot].next();
            }
        }
    }

    pub fn update(&mut self, dt: f32) {
        for c in &mut self.cooldown {
            *c = (*c - dt).max(0.0);
        }
        if let Some(a) = &mut self.active {
            a.elapsed += dt;
            if a.elapsed >= a.kind.def().duration {
                self.cooldown[a.slot] = a.kind.def().cooldown;
                self.active = None;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn costs_scale_with_mass_with_a_floor() {
        assert_eq!(FormKind::Spear.cost(10), 5);
        assert_eq!(FormKind::Spear.cost(1_000), 40);
        assert_eq!(FormKind::Bloom.cost(1_000), 80);
    }

    #[test]
    fn activation_duration_and_cooldown() {
        let mut f = Forms::default();
        assert_eq!(f.try_activate(0, 100, Vec2::X), Ok(5));
        assert_eq!(f.try_activate(1, 100, Vec2::X), Err(FormError::Busy));
        // Spear holds 2.5 s, then cools down for 6 s.
        for _ in 0..160 {
            f.update(1.0 / 60.0);
        }
        assert!(f.active.is_none());
        assert_eq!(f.try_activate(0, 100, Vec2::X), Err(FormError::Cooling));
        assert!(
            f.try_activate(1, 100, Vec2::X).is_ok(),
            "slot 2 is independent"
        );
        for _ in 0..(10.0 * 60.0) as usize {
            f.update(1.0 / 60.0);
        }
        assert!(f.try_activate(0, 100, Vec2::X).is_ok());
    }

    #[test]
    fn a_tiny_swarm_cannot_spend_itself() {
        let mut f = Forms::default();
        assert_eq!(f.try_activate(1, 8, Vec2::X), Err(FormError::TooSmall));
    }

    #[test]
    fn cycling_skips_the_other_slot() {
        let mut f = Forms::default();
        f.cycle(0); // Spear -> Shield is taken by slot 2 -> Ring
        assert_eq!(f.slots, [FormKind::Ring, FormKind::Shield]);
    }
}
