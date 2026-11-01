//! Level files: an ASCII map of tiles, expanded into a material grid plus the
//! spots gameplay needs (Queen start, hazard route, food clusters).
//!
//! See `assets/levels/kitchen.txt` for the format.

use anyhow::{Context, Result, bail, ensure};
use glam::Vec2;

use crate::sim::WorldBounds;
use crate::sim::material::{CELLS_PER_UNIT, FULL, Kind, MaterialLayout};
use crate::util::SplitMix64;

pub const KITCHEN: &str = include_str!("../assets/levels/kitchen.txt");

#[derive(Clone, Debug)]
pub struct Level {
    pub material: MaterialLayout,
    pub queen_start: Vec2,
    /// Patrol route for the prototype hazard (the mop), in order.
    pub hazard_route: Vec<Vec2>,
    /// Centres of food clusters, for scripted play and AI.
    pub food_spots: Vec<Vec2>,
}

impl Level {
    pub fn kitchen(seed: u64) -> Self {
        Self::parse(KITCHEN, seed).expect("built-in kitchen level is valid")
    }

    pub fn bounds(&self) -> WorldBounds {
        self.material.bounds
    }

    pub fn parse(text: &str, seed: u64) -> Result<Self> {
        let mut tile = None;
        let mut rows: Vec<&str> = Vec::new();
        for line in text.lines() {
            if line.starts_with("# ") || line.trim().is_empty() {
                continue;
            }
            if let Some(v) = line.strip_prefix("tile ") {
                tile = Some(v.trim().parse::<f32>().context("bad tile size")?);
                continue;
            }
            rows.push(line.trim_end());
        }
        let tile = tile.context("missing `tile N` line")?;
        ensure!(!rows.is_empty(), "level has no map rows");
        let width = rows[0].chars().count();
        ensure!(
            rows.iter().all(|r| r.chars().count() == width),
            "all map rows must be the same width"
        );
        let height = rows.len();
        let size = Vec2::new(width as f32, height as f32) * tile;
        let bounds = WorldBounds {
            min: -size * 0.5,
            max: size * 0.5,
        };
        let mut material = MaterialLayout::empty(bounds);
        let cells_per_tile = (tile * CELLS_PER_UNIT).round() as u32;
        let mut rng = SplitMix64::new(seed);
        let mut queen_start = None;
        let mut route: Vec<(u32, Vec2)> = Vec::new();
        let mut grid: Vec<Vec<char>> = Vec::with_capacity(height);

        for (row, line) in rows.iter().enumerate() {
            grid.push(line.chars().collect());
            for (col, ch) in line.chars().enumerate() {
                // Row 0 is the top (north) edge, i.e. maximum y.
                let tile_min = Vec2::new(
                    bounds.min.x + col as f32 * tile,
                    bounds.max.y - (row + 1) as f32 * tile,
                );
                let center = tile_min + Vec2::splat(tile * 0.5);
                let kind = match ch {
                    '.' | 'Q' | '1'..='9' => Kind::Floor,
                    '#' | 'o' => Kind::Wall,
                    'c' => Kind::Crumbs,
                    'j' => Kind::Jam,
                    'b' => Kind::Bread,
                    'k' => Kind::Cereal,
                    'h' => Kind::Cheese,
                    'w' => Kind::Water,
                    other => bail!("unknown map character `{other}` at row {row}, column {col}"),
                };
                match ch {
                    'Q' => queen_start = Some(center),
                    '1'..='9' => route.push((ch.to_digit(10).unwrap_or(0), center)),
                    _ => {}
                }
                let cx0 = col as u32 * cells_per_tile;
                let cy0 = (height - 1 - row) as u32 * cells_per_tile;
                for dy in 0..cells_per_tile {
                    for dx in 0..cells_per_tile {
                        let i = ((cy0 + dy) * material.dim[0] + cx0 + dx) as usize;
                        material.kinds[i] = kind as u32;
                        material.amounts[i] = match kind {
                            Kind::Floor | Kind::Wall | Kind::Water => 0,
                            // Crumbs are sparse specks of varying size.
                            Kind::Crumbs => {
                                if rng.next_f32() < 0.3 {
                                    (FULL as f32 * (0.4 + 0.6 * rng.next_f32())) as u32
                                } else {
                                    0
                                }
                            }
                            Kind::Jam => (FULL as f32 * 0.8) as u32,
                            Kind::Bread | Kind::Cereal | Kind::Cheese => FULL,
                        };
                    }
                }
            }
        }
        route.sort_by_key(|&(n, _)| n);
        Ok(Self {
            material,
            queen_start: queen_start.context("level has no `Q` (Queen start)")?,
            hazard_route: route.into_iter().map(|(_, p)| p).collect(),
            food_spots: food_clusters(&grid, tile, bounds),
        })
    }
}

/// Centres of 8-connected clusters of food tiles.
fn food_clusters(grid: &[Vec<char>], tile: f32, bounds: WorldBounds) -> Vec<Vec2> {
    let is_food = |c: char| matches!(c, 'c' | 'j' | 'b' | 'k' | 'h');
    let h = grid.len();
    let w = grid.first().map_or(0, Vec::len);
    let mut seen = vec![vec![false; w]; h];
    let mut spots = Vec::new();
    for r0 in 0..h {
        for c0 in 0..w {
            if seen[r0][c0] || !is_food(grid[r0][c0]) {
                continue;
            }
            let mut stack = vec![(r0, c0)];
            seen[r0][c0] = true;
            let (mut sum, mut n) = (Vec2::ZERO, 0.0);
            while let Some((r, c)) = stack.pop() {
                sum += Vec2::new(
                    bounds.min.x + (c as f32 + 0.5) * tile,
                    bounds.max.y - (r as f32 + 0.5) * tile,
                );
                n += 1.0;
                for dr in -1i32..=1 {
                    for dc in -1i32..=1 {
                        let (rr, cc) = (r as i32 + dr, c as i32 + dc);
                        if rr < 0 || cc < 0 || rr >= h as i32 || cc >= w as i32 {
                            continue;
                        }
                        let (rr, cc) = (rr as usize, cc as usize);
                        if !seen[rr][cc] && is_food(grid[rr][cc]) {
                            seen[rr][cc] = true;
                            stack.push((rr, cc));
                        }
                    }
                }
            }
            spots.push(sum / n);
        }
    }
    spots
}

/// Orders `spots` into a greedy nearest-neighbour tour starting at `start`.
pub fn tour(start: Vec2, spots: &[Vec2]) -> Vec<Vec2> {
    let mut left: Vec<Vec2> = spots.to_vec();
    let mut out = Vec::with_capacity(left.len());
    let mut at = start;
    while !left.is_empty() {
        let (i, _) = left
            .iter()
            .enumerate()
            .min_by(|a, b| {
                a.1.distance_squared(at)
                    .total_cmp(&b.1.distance_squared(at))
            })
            .expect("non-empty");
        at = left.swap_remove(i);
        out.push(at);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const TINY: &str = "# test\ntile 2\n#####\n#.b1#\n#Q.c#\n#####\n";

    #[test]
    fn parses_a_tiny_level() {
        let level = Level::parse(TINY, 1).unwrap();
        let m = &level.material;
        // 5 x 4 tiles of 2 units, 2 cells per unit.
        assert_eq!(m.dim, [20, 16]);
        assert_eq!(level.bounds().size(), Vec2::new(10.0, 8.0));
        // Q is column 1, row 2 (counting from the top): x = -5 + 3, y = 4 - 5.
        assert_eq!(level.queen_start, Vec2::new(-2.0, -1.0));
        assert_eq!(level.hazard_route, vec![Vec2::new(2.0, 1.0)]);
        // The bread tile is solid food.
        let bread = m.index_of(Vec2::new(0.0, 1.0)).unwrap();
        assert_eq!(m.kinds[bread], Kind::Bread as u32);
        assert_eq!(m.amounts[bread], FULL);
        assert!(m.is_wall(Vec2::new(-4.5, 0.0)));
        assert_eq!(
            level.food_spots.len(),
            1,
            "b and c touch diagonally: one cluster"
        );
    }

    #[test]
    fn rejects_bad_levels() {
        assert!(Level::parse("tile 2\n#x#\n", 1).is_err());
        assert!(Level::parse("tile 2\n###\n##\n", 1).is_err());
        assert!(Level::parse("#Q#\n", 1).is_err(), "missing tile line");
        assert!(Level::parse("tile 2\n#.#\n", 1).is_err(), "missing Q");
    }

    #[test]
    fn kitchen_loads_with_food_and_a_route() {
        let level = Level::kitchen(1);
        assert_eq!(level.bounds().size(), Vec2::new(240.0, 160.0));
        assert!(level.hazard_route.len() >= 3);
        assert!(level.food_spots.len() >= 5);
        assert!(level.material.total_food() > 1_000 * FULL as u64);
        assert!(!level.material.is_wall(level.queen_start));
    }

    #[test]
    fn tour_visits_everything_nearest_first() {
        let spots = [
            Vec2::new(10.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(5.0, 0.0),
        ];
        let t = tour(Vec2::ZERO, &spots);
        assert_eq!(
            t,
            vec![
                Vec2::new(1.0, 0.0),
                Vec2::new(5.0, 0.0),
                Vec2::new(10.0, 0.0)
            ]
        );
    }
}
