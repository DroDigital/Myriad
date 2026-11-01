// Shared declarations for every swarm compute pass.
// Must match `SimParams` and `Agent` in src/sim/mod.rs byte for byte.
// The material kind table (KIND_*, FULL, FLAG_*) is generated from Rust and
// prepended before this file (see `material::wgsl_kind_table`).

struct Params {
    focal: vec2f,
    world_min: vec2f,
    world_max: vec2f,
    grid_dim: vec2u,
    cell_size: f32,
    dt: f32,
    capacity: u32,
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
    eat_rate: f32,
    hatch_count: u32,
    queen_pos: vec2f,
    queen_radius: f32,
    guard_radius: f32,
    hazard_pos: vec2f,
    hazard_radius: f32,
    hazard_active: u32,
    mat_dim: vec2u,
    mat_cell: f32,
    step: u32,
    forage_strength: f32,
    forage_radius: f32,
    // Critters within this distance of the focal point form the swarm's
    // "main body" (what the Queen and camera follow).
    body_radius: f32,
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

// Indices into the `counters` buffer. Must match `counter` in src/sim/mod.rs.
const LIVE_END: u32 = 0u;      // agents in [0, LIVE_END) may be alive
const HATCHED: u32 = 1u;       // cumulative hatchlings placed
const EATEN: u32 = 2u;         // cumulative food eaten, fixed point (FULL = 1 cell)
const DIED: u32 = 3u;          // cumulative critters killed
const TICKETS: u32 = 4u;       // per step: dead slots that claimed a hatch ticket

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

// A newborn critter in slot `i`, placed in a small disk around the Queen.
fn hatchling(i: u32) -> Agent {
    let h = hash_u32(i * 2654435761u ^ hash_u32(params.step));
    let ang = hash_f(h) * 6.2831853;
    let r = params.queen_radius * (0.6 + 1.2 * sqrt(hash_f(h ^ 0x9e3779b9u)));
    var a: Agent;
    a.pos = params.queen_pos + vec2f(cos(ang), sin(ang)) * r;
    a.vel = vec2f(cos(ang), sin(ang)) * 2.0;
    a.seed = hash_f(h ^ 0x85ebca6bu);
    a.flags = ALIVE;
    a.caste = 0u;
    a.crowding = 0.0;
    return a;
}
