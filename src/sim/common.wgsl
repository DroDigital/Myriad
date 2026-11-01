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
    // Path-cost multiplier for chewing through solid food (flow field).
    solid_cost: f32,
    // Active Form (0 = none; see `forms.rs`), its progress 0..1, where it is
    // centred, which way it points, and its size.
    form_origin: vec2f,
    form_dir: vec2f,
    form_kind: u32,
    form_t: f32,
    form_size: f32,
    // Non-zero while the Build button is held.
    build_active: u32,
    // Critters to give up this step (mass spent beyond the bank).
    sacrifice: u32,
    // Chance per step that a critter on open water drowns.
    drown_rate: f32,
    // Non-zero on the step that merges pairs of agents (act transitions).
    merge: u32,
    // Unscaled rest density: how crowded counts as "crowded" for spray, so
    // Gather (which packs tighter) is riskier under fire than Spread.
    base_rest_density: f32,
}

// One enemy (or one extra danger zone of a boss). Must match `Threat` in
// src/sim/mod.rs. `body_radius` 0 = no body; `zone_kind` 0 = no zone.
struct Threat {
    pos: vec2f,
    dir: vec2f,
    body_radius: f32,
    zone_kind: u32,
    zone_len: f32,
    // Circle: radius. Cone: half-angle (radians). Lane: half-width.
    zone_width: f32,
    // Chance per step that a critter in the zone dies.
    lethality: f32,
    // Velocity multiplier in the zone.
    slow: f32,
    // 0 off, 1 telegraphing (harmless), 2 active.
    zone_on: u32,
    look: u32,
    health: f32,
    telegraph: f32,
    pad: vec2f,
}

struct Threats {
    items: array<Threat, 8>,
}

const MAX_THREATS: u32 = 8u;
const ZONE_CIRCLE: u32 = 1u;
const ZONE_CONE: u32 = 2u;
const ZONE_LANE: u32 = 3u;

// Is `p` inside threat `t`'s zone?
fn in_zone(t: Threat, p: vec2f) -> bool {
    let d = p - t.pos;
    switch t.zone_kind {
        case ZONE_CIRCLE: {
            return length(d) < t.zone_width;
        }
        case ZONE_CONE: {
            let len = length(d);
            return len < t.zone_len && len > 1e-3 && dot(d / len, t.dir) > cos(t.zone_width);
        }
        case ZONE_LANE: {
            let h = clamp(dot(d, t.dir), 0.0, t.zone_len);
            return length(d - t.dir * h) < t.zone_width;
        }
        default: {
            return false;
        }
    }
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
// Holding a bridge over water (only while Build is held).
const ANCHORED: u32 = 2u;
const INVALID: u32 = 0xffffffffu;

// Indices into the `counters` buffer. Must match `counter` in src/sim/mod.rs.
const LIVE_END: u32 = 0u;      // agents in [0, LIVE_END) may be alive
const HATCHED: u32 = 1u;       // cumulative hatchlings placed
const EATEN: u32 = 2u;         // cumulative food eaten, fixed point (FULL = 1 cell)
const DIED: u32 = 3u;          // cumulative critters killed
const TICKETS: u32 = 4u;       // per step: dead slots that claimed a hatch ticket
const SACRIFICED: u32 = 5u;    // cumulative critters spent on Forms
const SAC_TICKETS: u32 = 6u;   // per step: critters that claimed a sacrifice ticket
const MERGED: u32 = 7u;        // cumulative agents folded into a neighbour
const ENEMY_HITS: u32 = 8u;    // 8 slots: cumulative contact damage per threat

// Form kinds. Must match `FormKind` in src/forms.rs.
const FORM_NONE: u32 = 0u;
const FORM_SPEAR: u32 = 1u;
const FORM_SHIELD: u32 = 2u;
const FORM_RING: u32 = 3u;
const FORM_COMET: u32 = 4u;
const FORM_BLOOM: u32 = 5u;

const FLOW_INF: f32 = 1e9;

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
