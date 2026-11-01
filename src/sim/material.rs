//! The edible world: a 2D grid of material cells on the GPU.
//!
//! Each cell has a kind (floor, wall, bread…) and an amount of food left, in
//! fixed point ([`FULL`] = one full cell). Critters erode the amount with
//! atomic compare-exchange, so the total eaten always equals the total removed.
//! "Solid" kinds block movement until they have been eaten away, which is how
//! the swarm eats its way through things.

use glam::Vec2;

use super::WorldBounds;
use crate::gpu::Gpu;

/// Fixed-point amount of food in one full cell.
pub const FULL: u32 = 1000;
/// Material cells per world unit (a critter is about 0.45 units long).
pub const CELLS_PER_UNIT: f32 = 2.0;

/// Always blocks movement.
pub const FLAG_BLOCKED: u32 = 1;
/// Blocks movement while any food remains in the cell.
pub const FLAG_SOLID: u32 = 2;
/// Can be eaten.
pub const FLAG_EDIBLE: u32 = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Kind {
    Floor = 0,
    Wall = 1,
    Crumbs = 2,
    Bread = 3,
    Cereal = 4,
    Jam = 5,
    Cheese = 6,
}

#[derive(Clone, Copy, Debug)]
pub struct KindInfo {
    pub kind: Kind,
    pub flags: u32,
    /// Multiplier on how fast critters eat it.
    pub eat_rate: f32,
    /// Display colour (sRGB).
    pub color: [f32; 3],
}

pub const KINDS: [KindInfo; 7] = [
    KindInfo {
        kind: Kind::Floor,
        flags: 0,
        eat_rate: 0.0,
        color: [0.0, 0.0, 0.0],
    },
    KindInfo {
        kind: Kind::Wall,
        flags: FLAG_BLOCKED,
        eat_rate: 0.0,
        color: [0.42, 0.31, 0.22],
    },
    KindInfo {
        kind: Kind::Crumbs,
        flags: FLAG_EDIBLE,
        eat_rate: 1.5,
        color: [0.80, 0.60, 0.33],
    },
    KindInfo {
        kind: Kind::Bread,
        flags: FLAG_EDIBLE | FLAG_SOLID,
        eat_rate: 1.0,
        color: [0.85, 0.62, 0.34],
    },
    KindInfo {
        kind: Kind::Cereal,
        flags: FLAG_EDIBLE | FLAG_SOLID,
        eat_rate: 0.45,
        color: [0.86, 0.24, 0.18],
    },
    KindInfo {
        kind: Kind::Jam,
        flags: FLAG_EDIBLE,
        eat_rate: 2.0,
        color: [0.55, 0.06, 0.20],
    },
    KindInfo {
        kind: Kind::Cheese,
        flags: FLAG_EDIBLE | FLAG_SOLID,
        eat_rate: 0.8,
        color: [0.98, 0.80, 0.30],
    },
];

impl Kind {
    pub fn info(self) -> &'static KindInfo {
        &KINDS[self as usize]
    }
}

/// The kind table as WGSL constants, so Rust stays the single source of truth.
pub fn wgsl_kind_table() -> String {
    let n = KINDS.len();
    let flags: Vec<String> = KINDS.iter().map(|k| format!("{}u", k.flags)).collect();
    let rates: Vec<String> = KINDS.iter().map(|k| format!("{:?}", k.eat_rate)).collect();
    let colors: Vec<String> = KINDS
        .iter()
        .map(|k| {
            format!(
                "vec3f({:?}, {:?}, {:?})",
                k.color[0], k.color[1], k.color[2]
            )
        })
        .collect();
    format!(
        "const FULL: u32 = {FULL}u;\n\
         const FLAG_BLOCKED: u32 = {FLAG_BLOCKED}u;\n\
         const FLAG_SOLID: u32 = {FLAG_SOLID}u;\n\
         const FLAG_EDIBLE: u32 = {FLAG_EDIBLE}u;\n\
         const KIND_COUNT: u32 = {n}u;\n\
         const KIND_FLAGS = array<u32, {n}>({});\n\
         const KIND_RATE = array<f32, {n}>({});\n\
         const KIND_COLOR = array<vec3f, {n}>({});\n",
        flags.join(", "),
        rates.join(", "),
        colors.join(", "),
    )
}

/// CPU description of a material grid, ready to upload.
#[derive(Clone, Debug)]
pub struct MaterialLayout {
    pub dim: [u32; 2],
    /// Cell edge in world units.
    pub cell: f32,
    pub bounds: WorldBounds,
    pub kinds: Vec<u32>,
    pub amounts: Vec<u32>,
}

impl MaterialLayout {
    /// All floor, covering `bounds`.
    pub fn empty(bounds: WorldBounds) -> Self {
        let cell = 1.0 / CELLS_PER_UNIT;
        let size = bounds.size() / cell;
        let dim = [size.x.ceil() as u32, size.y.ceil() as u32];
        let n = (dim[0] * dim[1]) as usize;
        Self {
            dim,
            cell,
            bounds,
            kinds: vec![Kind::Floor as u32; n],
            amounts: vec![0; n],
        }
    }

    pub fn len(&self) -> usize {
        self.kinds.len()
    }

    pub fn is_empty(&self) -> bool {
        self.kinds.is_empty()
    }

    /// Cell index of a world position, or `None` outside the grid.
    pub fn index_of(&self, p: Vec2) -> Option<usize> {
        let rel = (p - self.bounds.min) / self.cell;
        if rel.x < 0.0 || rel.y < 0.0 {
            return None;
        }
        let (x, y) = (rel.x as u32, rel.y as u32);
        (x < self.dim[0] && y < self.dim[1]).then(|| (y * self.dim[0] + x) as usize)
    }

    /// World-space centre of cell `i`.
    pub fn center_of(&self, i: usize) -> Vec2 {
        let x = (i as u32 % self.dim[0]) as f32 + 0.5;
        let y = (i as u32 / self.dim[0]) as f32 + 0.5;
        self.bounds.min + Vec2::new(x, y) * self.cell
    }

    /// Fills every cell whose centre lies in the world-space rectangle.
    pub fn fill_rect(&mut self, min: Vec2, max: Vec2, kind: Kind, amount: u32) {
        for i in 0..self.len() {
            let c = self.center_of(i);
            if c.cmpge(min).all() && c.cmple(max).all() {
                self.kinds[i] = kind as u32;
                self.amounts[i] = amount;
            }
        }
    }

    /// Whether a position is permanently blocked (walls; ignores food).
    pub fn is_wall(&self, p: Vec2) -> bool {
        match self.index_of(p) {
            Some(i) => KINDS[self.kinds[i] as usize].flags & FLAG_BLOCKED != 0,
            None => true,
        }
    }

    /// Total food in fixed point.
    pub fn total_food(&self) -> u64 {
        self.amounts
            .iter()
            .zip(&self.kinds)
            .filter(|&(_, &k)| KINDS[k as usize].flags & FLAG_EDIBLE != 0)
            .map(|(&a, _)| a as u64)
            .sum()
    }
}

/// The material grid's GPU buffers.
pub struct MaterialGrid {
    pub dim: [u32; 2],
    pub cell: f32,
    pub bounds: WorldBounds,
    pub kinds: wgpu::Buffer,
    pub amounts: wgpu::Buffer,
    initial_food: u64,
}

impl MaterialGrid {
    pub fn new(gpu: &Gpu, layout: &MaterialLayout) -> Self {
        let make = |label: &str, data: &[u32]| {
            let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (data.len() * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            gpu.queue
                .write_buffer(&buffer, 0, bytemuck::cast_slice(data));
            buffer
        };
        Self {
            dim: layout.dim,
            cell: layout.cell,
            bounds: layout.bounds,
            kinds: make("material kinds", &layout.kinds),
            amounts: make("material amounts", &layout.amounts),
            initial_food: layout.total_food(),
        }
    }

    pub fn cell_count(&self) -> usize {
        (self.dim[0] * self.dim[1]) as usize
    }

    /// Food present at load time, in fixed point.
    pub fn initial_food(&self) -> u64 {
        self.initial_food
    }

    /// Reads all amounts back (blocking). For tests and tools.
    pub fn read_amounts(&self, gpu: &Gpu) -> anyhow::Result<Vec<u32>> {
        gpu.read_buffer(&self.amounts, self.cell_count())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bounds() -> WorldBounds {
        WorldBounds {
            min: Vec2::new(-10.0, -5.0),
            max: Vec2::new(10.0, 5.0),
        }
    }

    #[test]
    fn layout_indexing_round_trips() {
        let l = MaterialLayout::empty(bounds());
        assert_eq!(l.dim, [40, 20]);
        for i in [0, 39, 40, 799] {
            assert_eq!(l.index_of(l.center_of(i)), Some(i));
        }
        assert_eq!(l.index_of(Vec2::new(-10.1, 0.0)), None);
        assert!(l.is_wall(Vec2::new(11.0, 0.0)));
    }

    #[test]
    fn fill_rect_and_total_food() {
        let mut l = MaterialLayout::empty(bounds());
        l.fill_rect(Vec2::new(0.0, 0.0), Vec2::new(2.0, 1.0), Kind::Bread, FULL);
        // Centres at 0.25, 0.75, ... inside [0, 2] x [0, 1] -> 4 x 2 cells.
        assert_eq!(l.total_food(), 8 * FULL as u64);
        l.fill_rect(Vec2::new(-5.0, -5.0), Vec2::new(-4.0, 5.0), Kind::Wall, 0);
        assert!(l.is_wall(Vec2::new(-4.5, 0.0)));
        assert!(!l.is_wall(Vec2::new(0.5, 0.5)));
    }

    #[test]
    fn wgsl_table_mentions_every_kind() {
        let t = wgsl_kind_table();
        assert!(t.contains(&format!("array<u32, {}>", KINDS.len())));
        assert!(t.contains("const FULL: u32 = 1000u;"));
        for (i, k) in KINDS.iter().enumerate() {
            assert_eq!(k.kind as usize, i, "KINDS must be indexed by kind");
        }
    }
}
