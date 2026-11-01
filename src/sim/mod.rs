//! GPU swarm simulation: spatial-hash flocking toward the player's focal point.
//!
//! Each step runs four compute passes (see `passes.wgsl`):
//! 1. **count**: bucket living agents into grid cells (atomic per-cell counters)
//! 2. **scan**: exclusive prefix sum of the counters into cell start offsets
//! 3. **scatter**: write agent indices into a list sorted by cell
//! 4. **steer**: each agent samples its 3x3 neighbourhood (an equal, unbiased
//!    budget per cell), steers, integrates
//!
//! 5. **reduce**: per-workgroup sums, read back asynchronously (see [`feedback`])
//!
//! Gameplay never reads the agents back per frame: it gets a small summary one
//! or two frames late through [`Swarm::summary`]. [`Swarm::read_agents`] blocks
//! and exists for tests, stats and tooling.

pub mod feedback;
pub mod stats;

use bytemuck::{Pod, Zeroable};
use glam::Vec2;

use crate::gpu::Gpu;
use crate::util::SplitMix64;
use feedback::{Feedback, SwarmSummary};

pub const WORKGROUP_SIZE: u32 = 256;
pub const AGENT_ALIVE: u32 = 1;
pub const INVALID_CELL: u32 = u32::MAX;

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
    pub agent_count: u32,
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

/// Per-step inputs from gameplay.
#[derive(Clone, Copy, Debug)]
pub struct SimFrame {
    pub focal: Vec2,
    /// -1 = full Spread, 0 = neutral, +1 = full Gather.
    pub gather: f32,
    pub dt: f32,
    pub time: f32,
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

pub struct Swarm {
    config: SwarmConfig,
    grid_dim: [u32; 2],
    count: u32,
    pub tuning: SwarmTuning,
    params: wgpu::Buffer,
    agents: wgpu::Buffer,
    agents_next: wgpu::Buffer,
    counts: wgpu::Buffer,
    starts: wgpu::Buffer,
    agent_cell: wgpu::Buffer,
    sorted: wgpu::Buffer,
    partials: wgpu::Buffer,
    passes: [Pass; 5],
    feedback: Feedback,
    steps: u64,
}

impl Swarm {
    pub fn new(device: &wgpu::Device, config: SwarmConfig) -> Self {
        assert!(config.capacity > 0, "swarm capacity must be positive");
        assert!(config.cell_size > 0.0, "cell size must be positive");
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
        let counts = storage("cell counts", cell_count as u64 * 4);
        let starts = storage("cell starts", cell_count as u64 * 4);
        let agent_cell = storage("agent cell", cap * 4);
        let agent_local = storage("agent local", cap * 4);
        let sorted = storage("sorted agents", cap * 4);
        let partials_bytes = config.capacity.div_ceil(WORKGROUP_SIZE) as u64 * 16;
        let partials = storage("reduce partials", partials_bytes);

        let passes = [
            make_pass(
                device,
                "count",
                &[&params, &agents, &counts, &agent_cell, &agent_local],
            ),
            make_pass(device, "scan", &[&params, &counts, &starts]),
            make_pass(
                device,
                "scatter",
                &[&params, &agent_cell, &agent_local, &starts, &sorted],
            ),
            make_pass(
                device,
                "steer",
                &[&params, &agents, &agents_next, &counts, &starts, &sorted],
            ),
            make_pass(device, "reduce", &[&params, &agents, &partials]),
        ];
        let feedback = Feedback::new(device, partials_bytes);

        Self {
            config,
            grid_dim,
            count: 0,
            tuning: SwarmTuning::default(),
            params,
            agents,
            agents_next,
            counts,
            starts,
            agent_cell,
            sorted,
            partials,
            passes,
            feedback,
            steps: 0,
        }
    }

    pub fn count(&self) -> u32 {
        self.count
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

    /// The agent storage buffer, for rendering.
    pub fn agents_buffer(&self) -> &wgpu::Buffer {
        &self.agents
    }

    /// Appends agents, up to capacity. Returns how many were added.
    pub fn spawn(&mut self, queue: &wgpu::Queue, new_agents: &[Agent]) -> usize {
        let room = (self.config.capacity - self.count) as usize;
        let n = new_agents.len().min(room);
        if n > 0 {
            let offset = self.count as u64 * size_of::<Agent>() as u64;
            queue.write_buffer(&self.agents, offset, bytemuck::cast_slice(&new_agents[..n]));
            self.count += n as u32;
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
        SimParams {
            focal: frame.focal.to_array(),
            world_min: self.config.bounds.min.to_array(),
            world_max: self.config.bounds.max.to_array(),
            grid_dim: self.grid_dim,
            cell_size: cell,
            dt: frame.dt,
            agent_count: self.count,
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
                .max(rest_radius(self.count, t.rest_density) * 0.8),
            // Gather packs critters tighter; Spread loosens them.
            pressure_strength: t.pressure_strength,
            rest_density: t.rest_density * (1.0 + 1.5 * gather) * (1.0 - 0.6 * spread),
            _pad: [0.0; 2],
        }
    }

    /// Runs one simulation step and submits it.
    pub fn step(&mut self, gpu: &Gpu, frame: &SimFrame) {
        if self.count == 0 {
            return;
        }
        self.steps += 1;
        let params = self.params_for(frame);
        gpu.queue
            .write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("swarm step"),
            });
        self.encode_step(&mut encoder);
        let bytes = self.partials_bytes();
        let slot = self
            .feedback
            .record_copy(&mut encoder, &self.partials, bytes, self.steps);
        gpu.queue.submit([encoder.finish()]);
        if let Some(slot) = slot {
            self.feedback.after_submit(slot, bytes);
        }
    }

    /// Steps run so far.
    pub fn steps(&self) -> u64 {
        self.steps
    }

    fn partials_bytes(&self) -> u64 {
        self.count.div_ceil(WORKGROUP_SIZE) as u64 * 16
    }

    /// Collects finished readbacks without blocking and returns the newest
    /// summary (usually one or two steps old). `None` until the first arrives.
    pub fn summary(&mut self, gpu: &Gpu) -> Option<SwarmSummary> {
        let bytes = self.partials_bytes();
        self.feedback.poll(&gpu.device, bytes)
    }

    fn encode_step(&self, encoder: &mut wgpu::CommandEncoder) {
        let agent_groups = self.count.div_ceil(WORKGROUP_SIZE);
        encoder.clear_buffer(&self.counts, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("swarm"),
                timestamp_writes: None,
            });
            let [count, scan, scatter, steer, _] = &self.passes;
            for (p, groups) in [
                (count, agent_groups),
                (scan, 1),
                (scatter, agent_groups),
                (steer, agent_groups),
            ] {
                pass.set_pipeline(&p.pipeline);
                pass.set_bind_group(0, &p.bind_group, &[]);
                pass.dispatch_workgroups(groups, 1, 1);
            }
        }
        encoder.copy_buffer_to_buffer(
            &self.agents_next,
            0,
            &self.agents,
            0,
            self.count as u64 * size_of::<Agent>() as u64,
        );
        // Reduce the new state for the CPU (after the copy back).
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("swarm reduce"),
            timestamp_writes: None,
        });
        let reduce = &self.passes[4];
        pass.set_pipeline(&reduce.pipeline);
        pass.set_bind_group(0, &reduce.bind_group, &[]);
        pass.dispatch_workgroups(agent_groups, 1, 1);
    }

    /// Reads all agents back (blocking). For tests and stats only.
    pub fn read_agents(&self, gpu: &Gpu) -> anyhow::Result<Vec<Agent>> {
        gpu.read_buffer(&self.agents, self.count as usize)
    }

    /// Reads the spatial hash from the last step back (blocking).
    pub fn read_grid(&self, gpu: &Gpu) -> anyhow::Result<GridSnapshot> {
        let cells = (self.grid_dim[0] * self.grid_dim[1]) as usize;
        let n = self.count as usize;
        Ok(GridSnapshot {
            counts: gpu.read_buffer(&self.counts, cells)?,
            starts: gpu.read_buffer(&self.starts, cells)?,
            sorted: gpu.read_buffer(&self.sorted, n)?,
            agent_cell: gpu.read_buffer(&self.agent_cell, n)?,
        })
    }

    /// CPU mirror of the shader's cell lookup.
    pub fn cell_of(&self, p: Vec2) -> u32 {
        let rel = (p - self.config.bounds.min) / self.config.cell_size;
        let max = Vec2::new(self.grid_dim[0] as f32 - 1.0, self.grid_dim[1] as f32 - 1.0);
        let c = rel.floor().clamp(Vec2::ZERO, max);
        c.y as u32 * self.grid_dim[0] + c.x as u32
    }
}

/// Radius of a disk holding `count` critters at `rest_density` per cell (cell = 1 unit).
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
    let source = format!("{COMMON_WGSL}\n{}", pass_source(name));
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
        assert_eq!(size_of::<SimParams>(), 112);
    }

    #[test]
    fn every_pass_section_exists() {
        for name in ["count", "scan", "scatter", "steer", "reduce"] {
            assert!(pass_source(name).contains("fn main"), "{name}");
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
