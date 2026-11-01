//! GPU swarm simulation: spatial-hash flocking toward the player's focal point,
//! eating the material grid, hatching at the Queen, dying to hazards.
//!
//! Each step runs these compute passes (see `passes.wgsl`):
//! 0. **hatch_revive / hatch_append** (when hatching): place newborns at the
//!    Queen, reusing dead slots first, then extending the live range
//! 1. **count**: bucket living agents into grid cells (atomic per-cell counters)
//! 2. **scan**: exclusive prefix sum of the counters into cell start offsets
//! 3. **scatter**: write agent indices into a list sorted by cell
//! 4. **steer**: each agent samples its 3x3 neighbourhood (an equal, unbiased
//!    budget per cell), steers, moves with collision, eats, may die
//! 5. **reduce**: per-workgroup sums, read back asynchronously (see [`feedback`])
//!
//! The GPU owns the population: a small `counters` buffer holds the live range
//! and cumulative hatched/eaten/died totals. The CPU asks for hatchlings and
//! learns what happened one or two frames later through [`Swarm::summary`];
//! it never blocks. [`Swarm::read_agents`] blocks and exists for tests and tools.

pub mod feedback;
pub mod material;
pub mod stats;

use bytemuck::{Pod, Zeroable};
use glam::Vec2;

use crate::gpu::Gpu;
use crate::util::SplitMix64;
use feedback::{Feedback, SwarmSummary};
use material::MaterialGrid;

pub const WORKGROUP_SIZE: u32 = 256;
pub const AGENT_ALIVE: u32 = 1;
/// Holding up a bridge over water (only while Build is held).
pub const AGENT_ANCHORED: u32 = 2;
/// Flow-field relaxation sweeps per step. The field's wavefront moves this
/// many cells per step, so it crosses the kitchen in well under a second.
pub const FLOW_SWEEPS: u32 = 6;
/// Path distance meaning "unreached or unreachable".
pub const FLOW_INF: f32 = 1e9;
pub const INVALID_CELL: u32 = u32::MAX;

/// Indices into the GPU `counters` buffer. Must match `common.wgsl`.
pub mod counter {
    pub const LIVE_END: usize = 0;
    pub const HATCHED: usize = 1;
    pub const EATEN: usize = 2;
    pub const DIED: usize = 3;
    pub const TICKETS: usize = 4;
    pub const SACRIFICED: usize = 5;
    pub const SAC_TICKETS: usize = 6;
    pub const COUNT: usize = 8;
}

const COMMON_WGSL: &str = include_str!("common.wgsl");
const PASSES_WGSL: &str = include_str!("passes.wgsl");

/// One simulated critter (or, at large scales, a group of critters).
/// Layout must match `Agent` in `common.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Agent {
    pub pos: [f32; 2],
    pub vel: [f32; 2],
    /// Stable per-agent random value in [0, 1), used for wander and tint.
    pub seed: f32,
    pub flags: u32,
    /// Reserved for castes (Step 5).
    pub caste: u32,
    /// How far above rest density this critter's neighbourhood was last step
    /// (0 = at or below rest). Written by the steer pass; neighbours read it.
    pub crowding: f32,
}

impl Agent {
    pub fn new(pos: Vec2, seed: f32) -> Self {
        Self {
            pos: pos.to_array(),
            seed,
            flags: AGENT_ALIVE,
            ..Self::default()
        }
    }

    pub fn pos(&self) -> Vec2 {
        Vec2::from_array(self.pos)
    }

    pub fn vel(&self) -> Vec2 {
        Vec2::from_array(self.vel)
    }

    pub fn is_alive(&self) -> bool {
        self.flags & AGENT_ALIVE != 0
    }
}

/// Uniform block for every pass. Layout must match `Params` in `common.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct SimParams {
    pub focal: [f32; 2],
    pub world_min: [f32; 2],
    pub world_max: [f32; 2],
    pub grid_dim: [u32; 2],
    pub cell_size: f32,
    pub dt: f32,
    pub capacity: u32,
    pub cell_count: u32,
    pub time: f32,
    pub gather: f32,
    pub max_speed: f32,
    pub seek_strength: f32,
    pub separation_radius: f32,
    pub separation_strength: f32,
    pub align_strength: f32,
    pub cohesion_strength: f32,
    pub damping: f32,
    pub neighbor_radius: f32,
    pub max_neighbors: u32,
    pub slow_radius: f32,
    pub pressure_strength: f32,
    pub rest_density: f32,
    pub eat_rate: f32,
    pub hatch_count: u32,
    pub queen_pos: [f32; 2],
    pub queen_radius: f32,
    pub guard_radius: f32,
    pub hazard_pos: [f32; 2],
    pub hazard_radius: f32,
    pub hazard_active: u32,
    pub mat_dim: [u32; 2],
    pub mat_cell: f32,
    pub step: u32,
    pub forage_strength: f32,
    pub forage_radius: f32,
    pub body_radius: f32,
    pub solid_cost: f32,
    pub form_origin: [f32; 2],
    pub form_dir: [f32; 2],
    pub form_kind: u32,
    pub form_t: f32,
    pub form_size: f32,
    pub build_active: u32,
    pub sacrifice: u32,
    pub drown_rate: f32,
    pub _pad: [f32; 2],
}

/// Feel knobs for the swarm. Tweaked by Gather/Spread each step.
#[derive(Clone, Copy, Debug)]
pub struct SwarmTuning {
    pub max_speed: f32,
    pub seek_strength: f32,
    pub separation_radius: f32,
    pub separation_strength: f32,
    pub align_strength: f32,
    pub cohesion_strength: f32,
    pub damping: f32,
    /// Neighbours examined per agent, split evenly over the 9 nearby cells.
    /// Bounds cost in dense crowds; sums are scaled up so the cap adds no bias.
    pub max_neighbors: u32,
    /// Distance from the focal point at which critters start to slow down.
    pub slow_radius: f32,
    /// Push away from crowds, scaled by how far above rest density they are.
    pub pressure_strength: f32,
    /// Comfortable critters per unit area; crowding above this builds pressure.
    pub rest_density: f32,
    /// Material cells a critter eats per second (before the kind's rate).
    pub eat_rate: f32,
    /// Critters within this distance of the Queen count as her guard.
    pub guard_radius: f32,
    /// Pull toward nearby food.
    pub forage_strength: f32,
    /// How far critters sense food, in world units.
    pub forage_radius: f32,
    /// Path cost of chewing through a cell of solid food, relative to walking
    /// across an open one.
    pub solid_cost: f32,
    /// Chance per step that a critter on open water drowns.
    pub drown_rate: f32,
}

impl Default for SwarmTuning {
    fn default() -> Self {
        Self {
            max_speed: 9.0,
            seek_strength: 4.0,
            separation_radius: 0.55,
            separation_strength: 30.0,
            align_strength: 1.5,
            cohesion_strength: 0.4,
            damping: 0.8,
            max_neighbors: 64,
            slow_radius: 4.0,
            pressure_strength: 200.0,
            rest_density: 2.5,
            eat_rate: 0.5,
            guard_radius: 4.0,
            forage_strength: 14.0,
            forage_radius: 3.0,
            solid_cost: 12.0,
            drown_rate: 0.02,
        }
    }
}

/// Axis-aligned playable area in world units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldBounds {
    pub min: Vec2,
    pub max: Vec2,
}

impl WorldBounds {
    pub fn contains(&self, p: Vec2) -> bool {
        p.cmpge(self.min).all() && p.cmple(self.max).all()
    }

    pub fn size(&self) -> Vec2 {
        self.max - self.min
    }
}

/// A circular danger zone that kills critters on contact.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HazardZone {
    pub center: Vec2,
    pub radius: f32,
}

/// The Form the swarm is holding this step (see `crate::forms`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FormShape {
    /// `FormKind as u32`.
    pub kind: u32,
    pub origin: Vec2,
    /// Unit aim direction.
    pub dir: Vec2,
    pub size: f32,
    /// Progress 0..1.
    pub t: f32,
}

/// Per-step inputs from gameplay.
#[derive(Clone, Copy, Debug)]
pub struct SimFrame {
    pub focal: Vec2,
    /// -1 = full Spread, 0 = neutral, +1 = full Gather.
    pub gather: f32,
    pub dt: f32,
    pub time: f32,
    /// Hatchlings to place at the Queen this step.
    pub hatch: u32,
    pub queen_pos: Vec2,
    /// The Queen's body: critters keep out of it; hatchlings appear around it.
    pub queen_radius: f32,
    pub hazard: Option<HazardZone>,
    pub form: Option<FormShape>,
    /// Build held: critters bumping into open water anchor and become a bridge.
    pub build: bool,
    /// Critters to give up this step (mass spent beyond the bank).
    pub sacrifice: u32,
}

impl SimFrame {
    /// A frame with no Queen, hazards or hatching: just steering.
    pub fn steer(focal: Vec2, dt: f32, time: f32) -> Self {
        Self {
            focal,
            gather: 0.0,
            dt,
            time,
            hatch: 0,
            queen_pos: Vec2::splat(f32::MAX),
            queen_radius: 1.0,
            hazard: None,
            form: None,
            build: false,
            sacrifice: 0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SwarmConfig {
    /// Maximum agents; buffers are allocated for this many.
    pub capacity: u32,
    pub bounds: WorldBounds,
    /// Grid cell edge. Also the neighbour radius, so 3x3 cells cover it.
    pub cell_size: f32,
}

/// CPU copy of the spatial hash, for tests and debugging.
#[derive(Debug)]
pub struct GridSnapshot {
    pub counts: Vec<u32>,
    pub starts: Vec<u32>,
    pub sorted: Vec<u32>,
    pub agent_cell: Vec<u32>,
}

struct Pass {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
}

struct Passes {
    hatch_revive: Pass,
    hatch_append: Pass,
    water_reset: Pass,
    count: Pass,
    scan: Pass,
    scatter: Pass,
    flow: Pass,
    steer: Pass,
    reduce: Pass,
}

pub struct Swarm {
    config: SwarmConfig,
    grid_dim: [u32; 2],
    mat_dim: [u32; 2],
    mat_cell: f32,
    /// Agents placed from the CPU before the first step.
    spawned: u32,
    /// Latest known living count (from feedback, or the initial spawn).
    alive_hint: u32,
    pub tuning: SwarmTuning,
    params: wgpu::Buffer,
    agents: wgpu::Buffer,
    agents_next: wgpu::Buffer,
    /// Per-cell counts, then per-cell start offsets.
    grid: wgpu::Buffer,
    flow: wgpu::Buffer,
    agent_cell: wgpu::Buffer,
    sorted: wgpu::Buffer,
    counters: wgpu::Buffer,
    partials: wgpu::Buffer,
    passes: Passes,
    feedback: Feedback,
    steps: u64,
}

impl Swarm {
    pub fn new(gpu: &Gpu, config: SwarmConfig, material: &MaterialGrid) -> Self {
        assert!(config.capacity > 0, "swarm capacity must be positive");
        assert!(config.cell_size > 0.0, "cell size must be positive");
        assert_eq!(
            material.bounds, config.bounds,
            "material grid and swarm must cover the same world"
        );
        let device = &gpu.device;
        let size = config.bounds.size();
        let grid_dim = [
            (size.x / config.cell_size).ceil().max(1.0) as u32,
            (size.y / config.cell_size).ceil().max(1.0) as u32,
        ];
        let cell_count = grid_dim[0] * grid_dim[1];

        let storage = |label: &str, bytes: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let cap = config.capacity as u64;
        let agent_bytes = cap * size_of::<Agent>() as u64;
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("swarm params"),
            size: size_of::<SimParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let agents = storage("agents", agent_bytes);
        let agents_next = storage("agents next", agent_bytes);
        let grid = storage("cell counts and starts", cell_count as u64 * 8);
        let flow = storage("flow field", cell_count as u64 * 4);
        gpu.queue.write_buffer(
            &flow,
            0,
            bytemuck::cast_slice(&vec![FLOW_INF; cell_count as usize]),
        );
        let agent_cell = storage("agent cell", cap * 4);
        let agent_local = storage("agent local", cap * 4);
        let sorted = storage("sorted agents", cap * 4);
        let counters = storage("counters", counter::COUNT as u64 * 4);
        let partials_bytes = config.capacity.div_ceil(WORKGROUP_SIZE) as u64 * 32;
        let partials = storage("reduce partials", partials_bytes);

        let mp = |name: &str, buffers: &[&wgpu::Buffer]| make_pass(device, name, buffers);
        let passes = Passes {
            hatch_revive: mp("hatch_revive", &[&params, &agents, &counters]),
            hatch_append: mp("hatch_append", &[&params, &agents, &counters]),
            water_reset: mp(
                "water_reset",
                &[&params, &material.amounts, &material.kinds],
            ),
            count: mp(
                "count",
                &[
                    &params,
                    &agents,
                    &grid,
                    &agent_cell,
                    &agent_local,
                    &counters,
                    &material.amounts,
                ],
            ),
            scan: mp("scan", &[&params, &grid]),
            scatter: mp(
                "scatter",
                &[
                    &params,
                    &agent_cell,
                    &agent_local,
                    &grid,
                    &sorted,
                    &counters,
                ],
            ),
            flow: mp(
                "flow",
                &[&params, &flow, &material.amounts, &material.kinds],
            ),
            steer: mp(
                "steer",
                &[
                    &params,
                    &agents,
                    &agents_next,
                    &grid,
                    &sorted,
                    &counters,
                    &material.amounts,
                    &material.kinds,
                    &flow,
                ],
            ),
            reduce: mp("reduce", &[&params, &agents, &partials, &counters]),
        };
        let feedback = Feedback::new(device, partials_bytes);

        Self {
            config,
            grid_dim,
            mat_dim: material.dim,
            mat_cell: material.cell,
            spawned: 0,
            alive_hint: 0,
            tuning: SwarmTuning::default(),
            params,
            agents,
            agents_next,
            grid,
            flow,
            agent_cell,
            sorted,
            counters,
            partials,
            passes,
            feedback,
            steps: 0,
        }
    }

    pub fn capacity(&self) -> u32 {
        self.config.capacity
    }

    pub fn bounds(&self) -> WorldBounds {
        self.config.bounds
    }

    pub fn grid_dim(&self) -> [u32; 2] {
        self.grid_dim
    }

    /// Latest known number of living critters (one or two frames old).
    pub fn alive_hint(&self) -> u32 {
        self.alive_hint
    }

    /// The agent storage buffer, for rendering. Draw `capacity()` instances;
    /// dead and unused slots render as nothing.
    pub fn agents_buffer(&self) -> &wgpu::Buffer {
        &self.agents
    }

    /// Places agents directly, up to capacity. Only valid before the first
    /// step; afterwards the GPU owns the population (use `SimFrame::hatch`).
    /// Returns how many were added.
    pub fn spawn(&mut self, queue: &wgpu::Queue, new_agents: &[Agent]) -> usize {
        assert_eq!(self.steps, 0, "spawn() is only valid before the first step");
        let room = (self.config.capacity - self.spawned) as usize;
        let n = new_agents.len().min(room);
        if n > 0 {
            let offset = self.spawned as u64 * size_of::<Agent>() as u64;
            queue.write_buffer(&self.agents, offset, bytemuck::cast_slice(&new_agents[..n]));
            self.spawned += n as u32;
            self.alive_hint = self.spawned;
            let live_end = (counter::LIVE_END * 4) as u64;
            queue.write_buffer(&self.counters, live_end, bytemuck::bytes_of(&self.spawned));
        }
        n
    }

    /// The uniform values for one step, with Gather/Spread applied.
    pub fn params_for(&self, frame: &SimFrame) -> SimParams {
        let t = &self.tuning;
        let g = frame.gather.clamp(-1.0, 1.0);
        let gather = g.max(0.0);
        let spread = (-g).max(0.0);
        let cell = self.config.cell_size;
        let hazard = frame.hazard.unwrap_or(HazardZone {
            center: Vec2::ZERO,
            radius: 0.0,
        });
        SimParams {
            focal: frame.focal.to_array(),
            world_min: self.config.bounds.min.to_array(),
            world_max: self.config.bounds.max.to_array(),
            grid_dim: self.grid_dim,
            cell_size: cell,
            dt: frame.dt,
            capacity: self.config.capacity,
            cell_count: self.grid_dim[0] * self.grid_dim[1],
            time: frame.time,
            gather: g,
            max_speed: t.max_speed,
            seek_strength: t.seek_strength * (1.0 + 0.8 * gather) * (1.0 - 0.5 * spread),
            // Separation must stay inside one cell for the 3x3 search to see it.
            separation_radius: (t.separation_radius * (1.0 - 0.35 * gather) * (1.0 + 0.7 * spread))
                .min(cell * 0.99),
            separation_strength: t.separation_strength,
            align_strength: t.align_strength,
            cohesion_strength: t.cohesion_strength * (1.0 + 2.0 * gather) * (1.0 - 0.8 * spread),
            damping: t.damping,
            neighbor_radius: cell,
            max_neighbors: t.max_neighbors,
            // Slow down within the swarm's natural radius, so it pools into a
            // disk instead of a crush at the focal point.
            slow_radius: t
                .slow_radius
                .max(rest_radius(self.alive_hint, t.rest_density) * 0.8),
            pressure_strength: t.pressure_strength,
            // Gather packs critters tighter; Spread loosens them.
            rest_density: t.rest_density
                * (1.0 + 1.5 * gather)
                * (1.0 - 0.6 * spread)
                // Shapes pack the swarm tight (Spear, Shield, Ring).
                * if frame.form.is_some_and(|f| (1..=3).contains(&f.kind)) {
                    2.0
                } else {
                    1.0
                },
            eat_rate: t.eat_rate,
            hatch_count: frame.hatch,
            queen_pos: frame.queen_pos.to_array(),
            queen_radius: frame.queen_radius,
            guard_radius: t.guard_radius,
            hazard_pos: hazard.center.to_array(),
            hazard_radius: hazard.radius,
            hazard_active: frame.hazard.is_some() as u32,
            mat_dim: self.mat_dim,
            mat_cell: self.mat_cell,
            step: self.steps as u32,
            forage_strength: t.forage_strength,
            forage_radius: t.forage_radius,
            body_radius: rest_radius(self.alive_hint, t.rest_density) * 2.0 + 8.0,
            solid_cost: t.solid_cost,
            form_origin: frame.form.map_or([0.0; 2], |f| f.origin.to_array()),
            form_dir: frame.form.map_or([1.0, 0.0], |f| f.dir.to_array()),
            form_kind: frame.form.map_or(0, |f| f.kind),
            form_t: frame.form.map_or(0.0, |f| f.t),
            form_size: frame.form.map_or(0.0, |f| f.size),
            build_active: frame.build as u32,
            sacrifice: frame.sacrifice,
            drown_rate: t.drown_rate,
            _pad: [0.0; 2],
        }
    }

    /// Runs one simulation step and submits it.
    pub fn step(&mut self, gpu: &Gpu, frame: &SimFrame) {
        self.steps += 1;
        let params = self.params_for(frame);
        gpu.queue
            .write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("swarm step"),
            });
        self.encode_step(&mut encoder, frame.hatch);
        let slot =
            self.feedback
                .record_copy(&mut encoder, &self.counters, &self.partials, self.steps);
        gpu.queue.submit([encoder.finish()]);
        if let Some(slot) = slot {
            self.feedback.after_submit(slot);
        }
    }

    /// Steps run so far.
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// Collects finished readbacks without blocking and returns the newest
    /// summary (usually one or two steps old). `None` until the first arrives.
    pub fn summary(&mut self, gpu: &Gpu) -> Option<SwarmSummary> {
        let s = self.feedback.poll(&gpu.device);
        if let Some(s) = s {
            self.alive_hint = s.alive;
        }
        s
    }

    fn encode_step(&self, encoder: &mut wgpu::CommandEncoder, hatch: u32) {
        let groups = self.config.capacity.div_ceil(WORKGROUP_SIZE);
        // Per-step ticket counters (TICKETS and SAC_TICKETS) start at zero.
        let tickets = (counter::TICKETS * 4) as u64;
        encoder.clear_buffer(&self.counters, tickets, Some(4));
        let sac = (counter::SAC_TICKETS * 4) as u64;
        encoder.clear_buffer(&self.counters, sac, Some(4));
        let cells = (self.grid_dim[0] * self.grid_dim[1]) as u64;
        encoder.clear_buffer(&self.grid, 0, Some(cells * 4));
        let mat_groups = (self.mat_dim[0] * self.mat_dim[1]).div_ceil(WORKGROUP_SIZE);
        let cell_groups = (cells as u32).div_ceil(WORKGROUP_SIZE);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("swarm"),
                timestamp_writes: None,
            });
            let p = &self.passes;
            let mut run = |pass_: &Pass, groups: u32| {
                pass.set_pipeline(&pass_.pipeline);
                pass.set_bind_group(0, &pass_.bind_group, &[]);
                pass.dispatch_workgroups(groups, 1, 1);
            };
            if hatch > 0 {
                run(&p.hatch_revive, groups);
                run(&p.hatch_append, hatch.div_ceil(WORKGROUP_SIZE));
            }
            run(&p.water_reset, mat_groups);
            run(&p.count, groups);
            run(&p.scan, 1);
            run(&p.scatter, groups);
            for _ in 0..FLOW_SWEEPS {
                run(&p.flow, cell_groups);
            }
            run(&p.steer, groups);
        }
        encoder.copy_buffer_to_buffer(&self.agents_next, 0, &self.agents, 0, None);
        // Reduce the new state for the CPU (after the copy back).
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("swarm reduce"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.passes.reduce.pipeline);
        pass.set_bind_group(0, &self.passes.reduce.bind_group, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }

    /// Reads every agent slot back (blocking). For tests and stats only;
    /// filter with [`Agent::is_alive`].
    pub fn read_agents(&self, gpu: &Gpu) -> anyhow::Result<Vec<Agent>> {
        gpu.read_buffer(&self.agents, self.config.capacity as usize)
    }

    /// Reads the counters back (blocking), indexed by [`counter`].
    pub fn read_counters(&self, gpu: &Gpu) -> anyhow::Result<Vec<u32>> {
        gpu.read_buffer(&self.counters, counter::COUNT)
    }

    /// Reads the spatial hash from the last step back (blocking).
    pub fn read_grid(&self, gpu: &Gpu) -> anyhow::Result<GridSnapshot> {
        let cells = (self.grid_dim[0] * self.grid_dim[1]) as usize;
        let n = self.config.capacity as usize;
        let mut grid: Vec<u32> = gpu.read_buffer(&self.grid, cells * 2)?;
        let starts = grid.split_off(cells);
        Ok(GridSnapshot {
            counts: grid,
            starts,
            sorted: gpu.read_buffer(&self.sorted, n)?,
            agent_cell: gpu.read_buffer(&self.agent_cell, n)?,
        })
    }

    /// Reads the flow field back (blocking): path distance to the focal point
    /// per grid cell, [`FLOW_INF`] where unreached. For tests and tools.
    pub fn read_flow(&self, gpu: &Gpu) -> anyhow::Result<Vec<f32>> {
        gpu.read_buffer(&self.flow, (self.grid_dim[0] * self.grid_dim[1]) as usize)
    }

    /// CPU mirror of the shader's cell lookup.
    pub fn cell_of(&self, p: Vec2) -> u32 {
        let rel = (p - self.config.bounds.min) / self.config.cell_size;
        let max = Vec2::new(self.grid_dim[0] as f32 - 1.0, self.grid_dim[1] as f32 - 1.0);
        let c = rel.floor().clamp(Vec2::ZERO, max);
        c.y as u32 * self.grid_dim[0] + c.x as u32
    }
}

/// Radius of a disk holding `count` critters at `rest_density` per unit area.
pub fn rest_radius(count: u32, rest_density: f32) -> f32 {
    (count as f32 / rest_density / std::f32::consts::PI).sqrt()
}

/// Returns the source of one `//! pass:<name>` section of `passes.wgsl`.
fn pass_source(name: &str) -> &'static str {
    let marker = format!("//! pass:{name}\n");
    let start = PASSES_WGSL
        .find(&marker)
        .unwrap_or_else(|| panic!("missing pass `{name}` in passes.wgsl"))
        + marker.len();
    let rest = &PASSES_WGSL[start..];
    let end = rest.find("//! pass:").unwrap_or(rest.len());
    &rest[..end]
}

fn make_pass(device: &wgpu::Device, name: &str, buffers: &[&wgpu::Buffer]) -> Pass {
    log::debug!("building pass `{name}`");
    let source = format!(
        "{}\n{COMMON_WGSL}\n{}",
        material::wgsl_kind_table(),
        pass_source(name)
    );
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(name),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(name),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let layout = pipeline.get_bind_group_layout(0);
    let entries: Vec<_> = buffers
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: b.as_entire_binding(),
        })
        .collect();
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(name),
        layout: &layout,
        entries: &entries,
    });
    Pass {
        pipeline,
        bind_group,
    }
}

/// Lays out `n` critters in a sunflower disk around `center`, about `spacing` apart.
pub fn spawn_disk(center: Vec2, n: usize, spacing: f32, rng: &mut SplitMix64) -> Vec<Agent> {
    const GOLDEN_ANGLE: f32 = 2.399_963_1;
    let c = spacing / std::f32::consts::PI.sqrt();
    (0..n)
        .map(|i| {
            let r = c * (i as f32 + 0.5).sqrt();
            let theta = i as f32 * GOLDEN_ANGLE;
            let jitter = Vec2::new(rng.next_f32() - 0.5, rng.next_f32() - 0.5) * spacing * 0.2;
            let pos = center + Vec2::new(theta.cos(), theta.sin()) * r + jitter;
            Agent::new(pos, rng.next_f32())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_struct_layouts_match_wgsl() {
        assert_eq!(size_of::<Agent>(), 32);
        assert_eq!(size_of::<SimParams>(), 224);
    }

    #[test]
    fn every_pass_section_exists() {
        for name in [
            "hatch_revive",
            "hatch_append",
            "water_reset",
            "count",
            "scan",
            "scatter",
            "flow",
            "steer",
            "reduce",
        ] {
            assert!(pass_source(name).contains("fn main"), "{name}");
        }
    }

    #[test]
    fn counter_indices_match_wgsl() {
        for (name, index) in [
            ("LIVE_END", counter::LIVE_END),
            ("HATCHED", counter::HATCHED),
            ("EATEN", counter::EATEN),
            ("DIED", counter::DIED),
            ("TICKETS", counter::TICKETS),
            ("SACRIFICED", counter::SACRIFICED),
            ("SAC_TICKETS", counter::SAC_TICKETS),
        ] {
            let decl = format!("const {name}: u32 = {index}u;");
            assert!(COMMON_WGSL.contains(&decl), "missing `{decl}`");
        }
    }

    #[test]
    fn spawn_disk_is_roughly_the_requested_density() {
        let mut rng = SplitMix64::new(1);
        let agents = spawn_disk(Vec2::new(5.0, -3.0), 1000, 0.5, &mut rng);
        assert_eq!(agents.len(), 1000);
        let max_r = agents
            .iter()
            .map(|a| (a.pos() - Vec2::new(5.0, -3.0)).length())
            .fold(0.0, f32::max);
        // Area of n critters at `spacing` is n * spacing^2 = 250 -> radius ~ 8.9.
        assert!((7.5..10.5).contains(&max_r), "radius {max_r}");
        assert!(agents.iter().all(Agent::is_alive));
    }
}
