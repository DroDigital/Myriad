//! CPU-side swarm statistics, computed from a readback. Used by tests and the
//! headless report, never per frame.

use std::fmt;

use glam::Vec2;

use super::{Agent, WorldBounds};

#[derive(Clone, Copy, Debug, Default)]
pub struct SwarmStats {
    pub alive: usize,
    pub centroid: Vec2,
    pub mean_speed: f32,
    pub mean_dist_to_focal: f32,
    /// Mean distance to the nearest neighbour, over a sample of agents.
    pub mean_nearest: f32,
    /// Agents with NaN or infinite position or velocity.
    pub non_finite: usize,
    pub out_of_bounds: usize,
}

impl SwarmStats {
    pub fn compute(agents: &[Agent], focal: Vec2, bounds: WorldBounds) -> Self {
        let alive: Vec<&Agent> = agents.iter().filter(|a| a.is_alive()).collect();
        let mut s = SwarmStats {
            alive: alive.len(),
            ..Default::default()
        };
        if alive.is_empty() {
            return s;
        }
        let mut finite = 0usize;
        // Allow a small epsilon: the shader clamps exactly to the bounds.
        let slack = Vec2::splat(1e-3);
        let loose = WorldBounds {
            min: bounds.min - slack,
            max: bounds.max + slack,
        };
        for a in &alive {
            let (p, v) = (a.pos(), a.vel());
            if !p.is_finite() || !v.is_finite() {
                s.non_finite += 1;
                continue;
            }
            if !loose.contains(p) {
                s.out_of_bounds += 1;
            }
            finite += 1;
            s.centroid += p;
            s.mean_speed += v.length();
            s.mean_dist_to_focal += (p - focal).length();
        }
        if finite > 0 {
            let inv = 1.0 / finite as f32;
            s.centroid *= inv;
            s.mean_speed *= inv;
            s.mean_dist_to_focal *= inv;
        }

        // Nearest-neighbour distance over an evenly spaced sample (brute force).
        let sample = alive.len().min(256);
        let stride = alive.len() / sample;
        let mut total = 0.0;
        for k in 0..sample {
            let i = k * stride;
            let p = alive[i].pos();
            let nearest = alive
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, b)| (b.pos() - p).length_squared())
                .fold(f32::INFINITY, f32::min);
            if nearest.is_finite() {
                total += nearest.sqrt();
            }
        }
        s.mean_nearest = total / sample as f32;
        s
    }
}

impl fmt::Display for SwarmStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "alive {} | centroid ({:.1}, {:.1}) | mean speed {:.2} | mean dist to focal {:.2} | \
             mean nearest {:.3} | non-finite {} | out of bounds {}",
            self.alive,
            self.centroid.x,
            self.centroid.y,
            self.mean_speed,
            self.mean_dist_to_focal,
            self.mean_nearest,
            self.non_finite,
            self.out_of_bounds
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_of_a_simple_line() {
        let agents: Vec<Agent> = (0..4)
            .map(|i| Agent::new(Vec2::new(i as f32, 0.0), 0.0))
            .collect();
        let bounds = WorldBounds {
            min: Vec2::splat(-10.0),
            max: Vec2::splat(10.0),
        };
        let s = SwarmStats::compute(&agents, Vec2::ZERO, bounds);
        assert_eq!(s.alive, 4);
        assert!((s.centroid.x - 1.5).abs() < 1e-5);
        assert!((s.mean_nearest - 1.0).abs() < 1e-5);
        assert_eq!(s.out_of_bounds, 0);
        assert_eq!(s.non_finite, 0);
    }
}
