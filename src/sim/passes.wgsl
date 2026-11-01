// Swarm passes. Each `//! pass:<name>` section is compiled as its own module
// (the kind table + common.wgsl prepended) so every pass gets an exact
// auto-derived layout. Order per step:
//   [hatch_revive -> hatch_append] -> water_reset -> count -> scan -> scatter
//   -> flow x N -> steer -> (copy) -> reduce
// The `grid` buffer holds per-cell counts in [0, cell_count) and cell start
// offsets in [cell_count, 2 * cell_count).
// Agent passes dispatch over the whole capacity and skip slots >= LIVE_END:
// the GPU owns the population, so the CPU never needs to know which slots
// are free.

//! pass:hatch_revive
// Dead slots below LIVE_END claim hatch tickets first, so slots get reused.
@group(0) @binding(1) var<storage, read_write> agents: array<Agent>;
@group(0) @binding(2) var<storage, read_write> counters: array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= atomicLoad(&counters[LIVE_END]) || (agents[i].flags & ALIVE) != 0u) {
        return;
    }
    let ticket = atomicAdd(&counters[TICKETS], 1u);
    if (ticket >= params.hatch_count) {
        return;
    }
    agents[i] = hatchling(i);
    atomicAdd(&counters[HATCHED], 1u);
}

//! pass:hatch_append
// Tickets no dead slot claimed extend the live range, up to capacity.
@group(0) @binding(1) var<storage, read_write> agents: array<Agent>;
@group(0) @binding(2) var<storage, read_write> counters: array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let t = gid.x;
    if (t >= params.hatch_count) {
        return;
    }
    let covered = min(atomicLoad(&counters[TICKETS]), params.hatch_count);
    if (t < covered) {
        return;
    }
    let idx = atomicAdd(&counters[LIVE_END], 1u);
    if (idx >= params.capacity) {
        atomicSub(&counters[LIVE_END], 1u);
        return;
    }
    agents[idx] = hatchling(idx);
    atomicAdd(&counters[HATCHED], 1u);
}

//! pass:count
// Bucket each living agent into its grid cell and reserve a slot in that cell.
@group(0) @binding(1) var<storage, read> agents: array<Agent>;
@group(0) @binding(2) var<storage, read_write> grid: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> agent_cell: array<u32>;
@group(0) @binding(4) var<storage, read_write> agent_local: array<u32>;
@group(0) @binding(5) var<storage, read> counters: array<u32>;
@group(0) @binding(6) var<storage, read_write> mat_amount: array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= counters[LIVE_END]) {
        return;
    }
    let a = agents[i];
    if ((a.flags & ALIVE) == 0u) {
        agent_cell[i] = INVALID;
        return;
    }
    let c = cell_index(cell_coord(a.pos));
    agent_cell[i] = c;
    agent_local[i] = atomicAdd(&grid[c], 1u);
    // An anchored critter holds up the water cell it stands in (water cells'
    // amount is their bridge count, reset each step by water_reset).
    if ((a.flags & ANCHORED) != 0u) {
        let rel = (a.pos - params.world_min) / params.mat_cell;
        let mc = vec2u(clamp(rel, vec2f(0.0), vec2f(params.mat_dim) - vec2f(1.0)));
        atomicAdd(&mat_amount[mc.y * params.mat_dim.x + mc.x], 1u);
    }
}

//! pass:scan
// Exclusive prefix sum of cell counts -> cell start offsets. One workgroup.
@group(0) @binding(1) var<storage, read_write> grid: array<u32>;

var<workgroup> partial: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32) {
    let n = params.cell_count;
    let chunk = (n + 255u) / 256u;
    let begin = t * chunk;
    let end = min(begin + chunk, n);
    var sum = 0u;
    for (var i = begin; i < end; i++) {
        sum += grid[i];
    }
    partial[t] = sum;
    workgroupBarrier();
    for (var offset = 1u; offset < 256u; offset *= 2u) {
        var v = 0u;
        if (t >= offset) {
            v = partial[t - offset];
        }
        workgroupBarrier();
        partial[t] += v;
        workgroupBarrier();
    }
    var running = partial[t] - sum;
    for (var i = begin; i < end; i++) {
        grid[n + i] = running;
        running += grid[i];
    }
}

//! pass:scatter
// Write each agent's index into its cell's slice of the sorted list.
@group(0) @binding(1) var<storage, read> agent_cell: array<u32>;
@group(0) @binding(2) var<storage, read> agent_local: array<u32>;
@group(0) @binding(3) var<storage, read> grid: array<u32>;
@group(0) @binding(4) var<storage, read_write> sorted: array<u32>;
@group(0) @binding(5) var<storage, read> counters: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= counters[LIVE_END]) {
        return;
    }
    let c = agent_cell[i];
    if (c >= params.cell_count) {
        return; // dead (INVALID) or out of range
    }
    sorted[grid[params.cell_count + c] + agent_local[i]] = i;
}

//! pass:water_reset
// Water cells store how many anchored critters hold them up this step; the
// count pass rebuilds it, so clear it first.
@group(0) @binding(1) var<storage, read_write> mat_amount: array<u32>;
@group(0) @binding(2) var<storage, read> mat_kind: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.mat_dim.x * params.mat_dim.y) {
        return;
    }
    var flags = KIND_FLAGS;
    if ((flags[min(mat_kind[i], KIND_COUNT - 1u)] & FLAG_WATER) != 0u) {
        mat_amount[i] = 0u;
    }
}

//! pass:flow
// One relaxation sweep of the flow field: each grid cell's path distance to
// the focal point is the cheapest neighbour's plus the cost of crossing this
// cell. Run several sweeps per step; it keeps up as the focal point moves and
// as food is eaten (solid food is costly, not impassable: the swarm can chew
// through when there is no way round).
@group(0) @binding(1) var<storage, read_write> flow: array<f32>;
@group(0) @binding(2) var<storage, read> mat_amount: array<u32>;
@group(0) @binding(3) var<storage, read> mat_kind: array<u32>;

fn crossing_cost(c: vec2u) -> f32 {
    let ratio = max(u32(round(params.cell_size / params.mat_cell)), 1u);
    var flags = KIND_FLAGS;
    var walls = 0.0;
    var solid = 0.0;
    var water = 0.0;
    for (var sy = 0u; sy < ratio; sy++) {
        for (var sx = 0u; sx < ratio; sx++) {
            let m = c * ratio + vec2u(sx, sy);
            if (m.x >= params.mat_dim.x || m.y >= params.mat_dim.y) {
                walls += 1.0;
                continue;
            }
            let mi = m.y * params.mat_dim.x + m.x;
            let f = flags[min(mat_kind[mi], KIND_COUNT - 1u)];
            let amount = mat_amount[mi];
            if ((f & FLAG_BLOCKED) != 0u) {
                walls += 1.0;
            } else if ((f & FLAG_SOLID) != 0u && amount > 0u) {
                solid += min(f32(amount) / f32(FULL), 1.0);
            } else if ((f & FLAG_WATER) != 0u && amount == 0u) {
                water += 1.0;
            }
        }
    }
    let n = f32(ratio * ratio);
    if (walls >= n) {
        return FLOW_INF;
    }
    var cost = 1.0 + 2.0 * walls / n + params.solid_cost * solid / n;
    if (water > 0.0) {
        // Open water is only crossable by building a bridge.
        if (params.build_active == 0u && water >= n) {
            return FLOW_INF;
        }
        cost += water / n * select(20.0, 3.0, params.build_active != 0u);
    }
    return cost;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.cell_count) {
        return;
    }
    let c = vec2u(i % params.grid_dim.x, i / params.grid_dim.x);
    if (i == cell_index(cell_coord(params.focal))) {
        flow[i] = 0.0;
        return;
    }
    let cost = crossing_cost(c);
    if (cost >= FLOW_INF) {
        flow[i] = FLOW_INF;
        return;
    }
    var best = FLOW_INF;
    let gd = vec2i(params.grid_dim);
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) {
                continue;
            }
            let nb = vec2i(c) + vec2i(dx, dy);
            if (nb.x < 0 || nb.y < 0 || nb.x >= gd.x || nb.y >= gd.y) {
                continue;
            }
            let step_len = select(1.0, 1.4142135, dx != 0 && dy != 0);
            best = min(best, flow[cell_index(vec2u(nb))] + cost * step_len);
        }
    }
    flow[i] = min(best, FLOW_INF);
}

//! pass:steer
// Flocking + crowd pressure + seek toward the focal point, then integrate with
// collision against the material grid, eat, and die to hazards. Reads
// `agents`, writes `agents_next` (copied back after the pass).
//
// Neighbour sums are estimated without bias: each of the 9 cells gets an equal
// budget, and a cell's sampled sum is scaled up by (others in cell / sampled).
// A shared cap scanned in order would see one side of a cell boundary more than
// the other and make critters pile up on the grid lines.
@group(0) @binding(1) var<storage, read> agents: array<Agent>;
@group(0) @binding(2) var<storage, read_write> agents_next: array<Agent>;
@group(0) @binding(3) var<storage, read> grid: array<u32>;
@group(0) @binding(4) var<storage, read> sorted: array<u32>;
@group(0) @binding(5) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> mat_amount: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read> mat_kind: array<u32>;
@group(0) @binding(8) var<storage, read> flow: array<f32>;

var<workgroup> wg_eaten: atomic<u32>;
var<workgroup> wg_died: atomic<u32>;

fn kind_flags(k: u32) -> u32 {
    var table = KIND_FLAGS;
    return table[min(k, KIND_COUNT - 1u)];
}

fn kind_rate(k: u32) -> f32 {
    var table = KIND_RATE;
    return table[min(k, KIND_COUNT - 1u)];
}

// Material cell at a world position, or -1 outside the grid.
fn mat_cell(p: vec2f) -> i32 {
    let rel = (p - params.world_min) / params.mat_cell;
    if (rel.x < 0.0 || rel.y < 0.0) {
        return -1;
    }
    let c = vec2u(rel);
    if (c.x >= params.mat_dim.x || c.y >= params.mat_dim.y) {
        return -1;
    }
    return i32(c.y * params.mat_dim.x + c.x);
}

fn blocked(p: vec2f) -> bool {
    let ci = mat_cell(p);
    if (ci < 0) {
        return true;
    }
    let f = kind_flags(mat_kind[ci]);
    if ((f & FLAG_BLOCKED) != 0u) {
        return true;
    }
    if ((f & FLAG_SOLID) != 0u) {
        return atomicLoad(&mat_amount[ci]) > 0u;
    }
    if ((f & FLAG_WATER) != 0u) {
        // Open water blocks; water held up by anchored critters is a bridge.
        return atomicLoad(&mat_amount[ci]) == 0u;
    }
    return false;
}

fn is_open_water(ci: i32) -> bool {
    return ci >= 0 && (kind_flags(mat_kind[ci]) & FLAG_WATER) != 0u
        && atomicLoad(&mat_amount[ci]) == 0u;
}

// Downhill direction of the flow field at `p` (xy) and the path distance
// there (z). z < 0 when the field has no answer yet (unreached or blocked).
fn flow_dir(p: vec2f) -> vec3f {
    let cc = vec2i(cell_coord(p));
    let gd = vec2i(params.grid_dim);
    let d0 = flow[cell_index(vec2u(cc))];
    if (d0 >= FLOW_INF * 0.5) {
        return vec3f(0.0, 0.0, -1.0);
    }
    var dir = vec2f(0.0);
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let nb = cc + vec2i(dx, dy);
            if ((dx == 0 && dy == 0) || nb.x < 0 || nb.y < 0 || nb.x >= gd.x || nb.y >= gd.y) {
                continue;
            }
            let off = vec2f(f32(dx), f32(dy));
            let drop = d0 - flow[cell_index(vec2u(nb))];
            if (drop > 0.0) {
                dir += off / dot(off, off) * drop;
            }
        }
    }
    if (length(dir) < 1e-5) {
        return vec3f(0.0, 0.0, -1.0);
    }
    return vec3f(normalize(dir), d0);
}

// Where the active Form wants this critter: xy = pull, z = 1 inside the Form.
fn form_pull(p: vec2f, seed: f32) -> vec3f {
    let o = params.form_origin;
    let size = params.form_size;
    switch params.form_kind {
        case FORM_SPEAR: {
            // A line from the swarm out along the aim. Each critter takes its
            // own spot along it (from its seed), which stretches the swarm out
            // instead of just sliding it forward.
            let tip = o + params.form_dir * size;
            let ab = tip - o;
            let slot = o + ab * seed;
            let to_slot = slot - p;
            let ds = max(length(to_slot), 1e-3);
            let h = clamp(dot(p - o, ab) / dot(ab, ab), 0.0, 1.0);
            let on_line = distance(p, o + ab * h) < max(1.0, size * 0.12) + 1.0;
            return vec3f(to_slot / ds * min(ds, 6.0) * 14.0, select(0.0, 1.0, on_line));
        }
        case FORM_SHIELD: {
            // A tight shell around the Queen.
            let d = p - o;
            let len = max(length(d), 1e-3);
            let q = o + d / len * size;
            // The band is thick enough to hold the whole swarm.
            let band = max(1.5, size * 0.5);
            return vec3f((q - p) * 8.0, select(0.0, 1.0, abs(len - size) < band));
        }
        case FORM_RING: {
            // A circle that closes in, crushing what is inside.
            let r = size * mix(1.0, 0.35, params.form_t);
            let d = p - o;
            let len = max(length(d), 1e-3);
            let q = o + d / len * r;
            return vec3f((q - p) * 8.0, select(0.0, 1.0, abs(len - r) < 1.5));
        }
        case FORM_COMET: {
            // A tight ball rushing along the aim direction.
            return vec3f((o - p) * 2.0, 1.0);
        }
        case FORM_BLOOM: {
            // A burst outward that fades.
            let d = p - o;
            let len = max(length(d), 1e-3);
            return vec3f(d / len * 45.0 * (1.0 - params.form_t), 1.0);
        }
        default: {
            return vec3f(0.0);
        }
    }
}

// How much faster critters inside the Form eat.
fn form_eat_mult(in_form: bool) -> f32 {
    if (!in_form) {
        return 1.0;
    }
    switch params.form_kind {
        case FORM_SPEAR: { return 4.0; }
        case FORM_RING: { return 3.0; }
        case FORM_BLOOM: { return 2.0; }
        default: { return 1.0; }
    }
}

// Takes up to `want` food from cell `ci`; returns what was actually taken.
// Compare-exchange keeps the books exact: eaten == removed, never negative.
fn bite(ci: i32, want: u32) -> u32 {
    if (want == 0u) {
        return 0u;
    }
    var cur = atomicLoad(&mat_amount[ci]);
    for (var tries = 0; tries < 8; tries++) {
        if (cur == 0u) {
            return 0u;
        }
        let take = min(cur, want);
        let r = atomicCompareExchangeWeak(&mat_amount[ci], cur, cur - take);
        if (r.exchanged) {
            return take;
        }
        cur = r.old_value;
    }
    return 0u;
}

// Unit-ish vector toward nearby food (zero if none in reach). Samples two
// rings of 8 directions, rotated per critter so the swarm doesn't streak.
fn forage(p: vec2f, seed: f32) -> vec2f {
    var pull = vec2f(0.0);
    var best = 0.0;
    for (var ring = 1; ring <= 2; ring++) {
        let r = params.forage_radius * f32(ring) * 0.5;
        for (var k = 0; k < 8; k++) {
            let ang = (f32(k) + seed) * 0.7853982;
            let dir = vec2f(cos(ang), sin(ang));
            let ci = mat_cell(p + dir * r);
            if (ci < 0) {
                continue;
            }
            let kind = mat_kind[ci];
            if ((kind_flags(kind) & FLAG_EDIBLE) == 0u) {
                continue;
            }
            let food = f32(atomicLoad(&mat_amount[ci])) / f32(FULL);
            // Nearer food counts for more.
            let w = food / f32(ring);
            pull += dir * w;
            best = max(best, w);
        }
    }
    let len = length(pull);
    if (len < 1e-4) {
        return vec2f(0.0);
    }
    return pull / len * min(best, 1.0);
}

fn bite_size(k: u32) -> u32 {
    return u32(params.eat_rate * kind_rate(k) * params.dt * f32(FULL) + 0.5);
}

// Simulates agent `i`; returns (food eaten, 1 if it died this step).
fn simulate(i: u32) -> vec2u {
    var a = agents[i];
    if ((a.flags & ALIVE) == 0u) {
        agents_next[i] = a;
        return vec2u(0u);
    }
    // Spend critters on Forms when the bank cannot cover the cost.
    if (params.sacrifice > 0u && atomicAdd(&counters[SAC_TICKETS], 1u) < params.sacrifice) {
        a.flags = 0u;
        agents_next[i] = a;
        atomicAdd(&counters[SACRIFICED], 1u);
        return vec2u(0u, 0u);
    }
    // Bridge builders hold still while Build is held, then let go.
    if ((a.flags & ANCHORED) != 0u) {
        if (params.build_active != 0u) {
            a.vel = vec2f(0.0);
            var died = 0u;
            if (params.hazard_active != 0u && distance(a.pos, params.hazard_pos) < params.hazard_radius) {
                a.flags = 0u;
                died = 1u;
            }
            agents_next[i] = a;
            return vec2u(0u, died);
        }
        a.flags &= ~ANCHORED;
    }

    let cc = vec2i(cell_coord(a.pos));
    let gd = vec2i(params.grid_dim);
    let nr = params.neighbor_radius;
    let sr = params.separation_radius;
    let budget = max(params.max_neighbors / 9u, 1u);
    let rot = hash_u32(i);

    var sep = vec2f(0.0);   // short-range: keep personal space
    var push = vec2f(0.0);  // long-range: flow away from crowds
    var push_c = vec2f(0.0); // same, weighted by each neighbour's crowding
    var dens = 0.0;         // kernel-weighted local density
    var vel_sum = vec2f(0.0);
    var pos_sum = vec2f(0.0);
    var n = 0.0;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let c = cc + vec2i(dx, dy);
            if (c.x < 0 || c.y < 0 || c.x >= gd.x || c.y >= gd.y) {
                continue;
            }
            let ci = cell_index(vec2u(c));
            let s = grid[params.cell_count + ci];
            let cnt = grid[ci];
            if (cnt == 0u) {
                continue;
            }
            let own = dx == 0 && dy == 0;
            let others = f32(cnt) - select(0.0, 1.0, own);
            let take = min(cnt, budget + select(0u, 1u, own));
            var c_sep = vec2f(0.0);
            var c_push = vec2f(0.0);
            var c_push_c = vec2f(0.0);
            var c_dens = 0.0;
            var c_vel = vec2f(0.0);
            var c_pos = vec2f(0.0);
            var c_n = 0.0;
            var seen = 0.0;
            for (var k = 0u; k < take; k++) {
                let j = sorted[s + (rot + k) % cnt];
                if (j == i) {
                    continue;
                }
                seen += 1.0;
                let o = agents[j];
                var d = a.pos - o.pos;
                let dist2 = dot(d, d);
                if (dist2 >= nr * nr) {
                    continue;
                }
                if (dist2 < 1e-10) {
                    // Coincident: push apart in a stable pseudo-random direction.
                    let ang = hash_f(i * 747796405u + j) * 6.2831853;
                    d = vec2f(cos(ang), sin(ang)) * 1e-3;
                }
                let dist = max(sqrt(dist2), 1e-3);
                let dir = d / dist;
                let w = 1.0 - dist / nr;
                if (dist < sr) {
                    c_sep += dir * (1.0 - dist / sr);
                }
                c_push += dir * w;
                c_push_c += dir * w * o.crowding;
                c_dens += w;
                c_vel += o.vel;
                c_pos += o.pos;
                c_n += 1.0;
            }
            if (seen > 0.0) {
                let scale = others / seen;
                sep += c_sep * scale;
                push += c_push * scale;
                push_c += c_push_c * scale;
                dens += c_dens * scale;
                vel_sum += c_vel * scale;
                pos_sum += c_pos * scale;
                n += c_n * scale;
            }
        }
    }

    // Crowding: 0 at or below rest density. For a uniform density rho the
    // kernel sum is rho * pi * nr^2 / 3.
    let rest_sum = params.rest_density * 1.0471976 * nr * nr;
    let crowding = max(dens / rest_sum - 1.0, 0.0);

    // Seek the focal point along the flow field (around walls, over bridges,
    // through food only when there is no way round), slowing on arrival so the
    // swarm pools around it. Close in, or where the field has no answer, head
    // straight for it.
    let to_focal = params.focal - a.pos;
    let df = length(to_focal);
    var desired = vec2f(0.0);
    let fd = flow_dir(a.pos);
    if (fd.z > 2.5) {
        desired = fd.xy * params.max_speed * smoothstep(0.0, params.slow_radius, fd.z);
    } else if (df > 1e-4) {
        desired = (to_focal / df) * params.max_speed * smoothstep(0.0, params.slow_radius, df);
    }
    // The active Form reshapes the swarm.
    let fp = form_pull(a.pos, a.seed);
    let in_form = fp.z > 0.5;
    var vmax_mult = 1.5;
    if (params.form_kind == FORM_SPEAR) {
        vmax_mult = 2.5;
    }
    if (params.form_kind == FORM_COMET) {
        desired = params.form_dir * params.max_speed * 2.5;
        vmax_mult = 3.0;
    } else if (params.form_kind == FORM_SHIELD) {
        // Turtle: stop advancing and hold the shell.
        desired = vec2f(0.0);
    } else if (params.form_kind != FORM_NONE && params.form_kind != FORM_BLOOM) {
        desired *= 0.3;
    }
    // Critters deep in a crowd shove a little less, so a big swarm flows
    // rather than crushing its own front.
    desired *= 1.0 / (1.0 + 0.5 * crowding);
    var acc = (desired - a.vel) * params.seek_strength;
    acc += sep * params.separation_strength;
    // Symmetric pressure: the push between two critters uses both of their
    // crowding values, so a crowded core pushes its uncrowded edge outward.
    acc += (push * crowding + push_c) * 0.5 / rest_sum * params.pressure_strength;
    if (n > 0.0) {
        let inv = 1.0 / n;
        acc += (vel_sum * inv - a.vel) * params.align_strength;
        acc += (pos_sum * inv - a.pos) * params.cohesion_strength;
    }
    // Forage: drift toward the richest food nearby, so the swarm flows over
    // whatever the player steers it onto instead of milling on bare floor.
    acc += forage(a.pos, a.seed) * params.forage_strength;
    acc += fp.xy;
    // The Queen's body: critters make room around her.
    let from_queen = a.pos - params.queen_pos;
    let dq = length(from_queen);
    if (dq < params.queen_radius && dq > 1e-4) {
        acc += (from_queen / dq) * (1.0 - dq / params.queen_radius) * 80.0;
    }
    // Per-critter wander keeps the swarm alive when it is at rest.
    let wander = a.seed * 6.2831853 + params.time * (0.7 + a.seed);
    acc += vec2f(cos(wander), sin(wander)) * 1.5;

    a.vel += acc * params.dt;
    a.vel *= 1.0 / (1.0 + params.damping * params.dt);
    let speed = length(a.vel);
    let vmax = params.max_speed * vmax_mult;
    if (speed > vmax) {
        a.vel *= vmax / speed;
    }

    // Move, sliding along walls and uneaten solids. A critter already inside
    // something (e.g. hatched over it) may move freely so it can get out.
    let old = a.pos;
    // Where the critter tried to go, before collision bounces its velocity.
    let intended = a.vel;
    let attempted = old + intended * params.dt;
    var np = attempted;
    var collided = false;
    if (blocked(np) && !blocked(old)) {
        collided = true;
        let px = vec2f(np.x, old.y);
        let py = vec2f(old.x, np.y);
        if (!blocked(px)) {
            np = px;
            a.vel.y *= -0.2;
        } else if (!blocked(py)) {
            np = py;
            a.vel.x *= -0.2;
        } else {
            np = old;
            a.vel *= -0.2;
        }
    }
    // Build: a critter that bumps into open water steps in and holds still,
    // becoming part of a bridge the rest can walk over.
    if (params.build_active != 0u && collided && !blocked(old)) {
        let wc = mat_cell(attempted);
        if (is_open_water(wc)) {
            np = attempted;
            a.vel = vec2f(0.0);
            a.flags |= ANCHORED;
        }
    }
    np = clamp(np, params.world_min, params.world_max);
    a.pos = np;
    a.crowding = crowding;

    // Eat: loose food underfoot, and solid food the critter is pushing into.
    var eaten = 0u;
    let here = mat_cell(np);
    if (here >= 0) {
        let k = mat_kind[here];
        let f = kind_flags(k);
        if ((f & FLAG_EDIBLE) != 0u && (f & FLAG_SOLID) == 0u) {
            eaten += bite(here, u32(f32(bite_size(k)) * form_eat_mult(in_form)));
        }
    }
    // Solid food: bite the cell it just bumped into, or else the one just
    // ahead along the direction it was trying to go (never the bounced-back
    // velocity, which would point away from the food it is pressed against).
    var gnaw = -1;
    if (collided) {
        gnaw = mat_cell(attempted);
    } else {
        var heading = intended;
        if (length(heading) < 1e-3) {
            heading = to_focal;
        }
        if (length(heading) > 1e-4) {
            gnaw = mat_cell(np + normalize(heading) * params.mat_cell * 0.9);
        }
    }
    if (gnaw >= 0 && gnaw != here) {
        let k = mat_kind[gnaw];
        let f = kind_flags(k);
        if ((f & FLAG_EDIBLE) != 0u && (f & FLAG_SOLID) != 0u) {
            eaten += bite(gnaw, u32(f32(bite_size(k)) * form_eat_mult(in_form)));
        }
    }

    // Hazards kill on contact, except critters holding the Shield. Critters
    // left on open water (a bridge let go) may drown.
    var died = 0u;
    let shielded = params.form_kind == FORM_SHIELD && in_form;
    if (params.hazard_active != 0u && !shielded
        && distance(np, params.hazard_pos) < params.hazard_radius) {
        died = 1u;
    }
    if ((a.flags & ANCHORED) == 0u && is_open_water(mat_cell(np))
        && hash_f(i ^ hash_u32(params.step * 0x9e3779b9u)) < params.drown_rate) {
        died = 1u;
    }
    if (died != 0u) {
        a.flags = 0u;
    }
    agents_next[i] = a;
    return vec2u(eaten, died);
}

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(local_invocation_index) t: u32,
) {
    let i = gid.x;
    var result = vec2u(0u);
    if (i < atomicLoad(&counters[LIVE_END])) {
        result = simulate(i);
    }
    if (result.x > 0u) {
        atomicAdd(&wg_eaten, result.x);
    }
    if (result.y > 0u) {
        atomicAdd(&wg_died, result.y);
    }
    // One global atomic per workgroup instead of one per critter.
    workgroupBarrier();
    if (t == 0u) {
        let e = atomicLoad(&wg_eaten);
        if (e > 0u) {
            atomicAdd(&counters[EATEN], e);
        }
        let d = atomicLoad(&wg_died);
        if (d > 0u) {
            atomicAdd(&counters[DIED], d);
        }
    }
}

//! pass:reduce
// Per-workgroup sums for the CPU to read back asynchronously: the swarm's
// state without stalling the GPU. Two vec4s per workgroup:
//   [0] = (main-body x sum, main-body y sum, main-body count, alive count)
//   [1] = (near the Queen, 0, 0, 0)
// The main body is the critters near the focal point, so stragglers stuck
// elsewhere do not drag the Queen and camera off the swarm.
@group(0) @binding(1) var<storage, read> agents: array<Agent>;
@group(0) @binding(2) var<storage, read_write> partials: array<vec4f>;
@group(0) @binding(3) var<storage, read> counters: array<u32>;

var<workgroup> acc_a: array<vec4f, 256>;
var<workgroup> acc_b: array<vec4f, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(local_invocation_index) t: u32,
    @builtin(workgroup_id) wg: vec3u,
) {
    var a4 = vec4f(0.0);
    var b4 = vec4f(0.0);
    if (gid.x < counters[LIVE_END]) {
        let a = agents[gid.x];
        if ((a.flags & ALIVE) != 0u) {
            let in_body = distance(a.pos, params.focal) < params.body_radius;
            a4 = select(vec4f(0.0, 0.0, 0.0, 1.0), vec4f(a.pos, 1.0, 1.0), in_body);
            b4.x = select(0.0, 1.0, distance(a.pos, params.queen_pos) < params.guard_radius);
        }
    }
    acc_a[t] = a4;
    acc_b[t] = b4;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (t < s) {
            acc_a[t] += acc_a[t + s];
            acc_b[t] += acc_b[t + s];
        }
        workgroupBarrier();
    }
    if (t == 0u) {
        partials[wg.x * 2u] = acc_a[0];
        partials[wg.x * 2u + 1u] = acc_b[0];
    }
}
