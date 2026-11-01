// Shared declarations for every swarm compute pass.
// Must match `SimParams` and `Agent` in src/sim/mod.rs byte for byte.

struct Params {
    focal: vec2f,
    world_min: vec2f,
    world_max: vec2f,
    grid_dim: vec2u,
    cell_size: f32,
    dt: f32,
    agent_count: u32,
    cell_count: u32,
    time: f32,
    gather: f32,
    max_speed: f32,
    seek_strength: f32,
    separation_radius: f32,
    separation_strength: f32,
    align_strength: f32,
    cohesion_strength: f32,
    damping: f32,
    neighbor_radius: f32,
    max_neighbors: u32,
    slow_radius: f32,
    pressure_strength: f32,
    rest_density: f32,
    pad0: f32,
    pad1: f32,
}

struct Agent {
    pos: vec2f,
    vel: vec2f,
    seed: f32,
    flags: u32,
    caste: u32,
    // How far above rest density this critter's neighbourhood was last step.
    crowding: f32,
}

const ALIVE: u32 = 1u;
const INVALID: u32 = 0xffffffffu;

@group(0) @binding(0) var<uniform> params: Params;

fn cell_coord(p: vec2f) -> vec2u {
    let rel = (p - params.world_min) / params.cell_size;
    let max_cell = vec2f(params.grid_dim) - vec2f(1.0);
    return vec2u(clamp(floor(rel), vec2f(0.0), max_cell));
}

fn cell_index(c: vec2u) -> u32 {
    return c.y * params.grid_dim.x + c.x;
}

fn hash_u32(x_in: u32) -> u32 {
    var x = x_in;
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

fn hash_f(x: u32) -> f32 {
    return f32(hash_u32(x)) / 4294967295.0;
}

