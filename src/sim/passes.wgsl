// Swarm passes. Each `//! pass:<name>` section is compiled as its own module
// (the kind table + common.wgsl prepended) so every pass gets an exact
// auto-derived layout. Order per step:
//   [hatch_revive -> hatch_append] -> count -> scan -> scatter -> steer -> (copy) -> reduce
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
@group(0) @binding(2) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> agent_cell: array<u32>;
@group(0) @binding(4) var<storage, read_write> agent_local: array<u32>;
@group(0) @binding(5) var<storage, read> counters: array<u32>;

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
    agent_local[i] = atomicAdd(&counts[c], 1u);
}

//! pass:scan
// Exclusive prefix sum of cell counts -> cell start offsets. One workgroup.
@group(0) @binding(1) var<storage, read> counts: array<u32>;
@group(0) @binding(2) var<storage, read_write> starts: array<u32>;

var<workgroup> partial: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32) {
    let n = params.cell_count;
    let chunk = (n + 255u) / 256u;
    let begin = t * chunk;
    let end = min(begin + chunk, n);
    var sum = 0u;
    for (var i = begin; i < end; i++) {
        sum += counts[i];
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
        starts[i] = running;
        running += counts[i];
    }
}

//! pass:scatter
// Write each agent's index into its cell's slice of the sorted list.
@group(0) @binding(1) var<storage, read> agent_cell: array<u32>;
@group(0) @binding(2) var<storage, read> agent_local: array<u32>;
@group(0) @binding(3) var<storage, read> starts: array<u32>;
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
    sorted[starts[c] + agent_local[i]] = i;
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
@group(0) @binding(3) var<storage, read> counts: array<u32>;
@group(0) @binding(4) var<storage, read> starts: array<u32>;
@group(0) @binding(5) var<storage, read> sorted: array<u32>;
@group(0) @binding(6) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> mat_amount: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read> mat_kind: array<u32>;

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
    return false;
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
            let s = starts[ci];
            let cnt = counts[ci];
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

    // Seek the focal point, slowing on arrival so the swarm pools around it.
    let to_focal = params.focal - a.pos;
    let df = length(to_focal);
    var desired = vec2f(0.0);
    if (df > 1e-4) {
        desired = (to_focal / df) * params.max_speed * smoothstep(0.0, params.slow_radius, df);
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
    let vmax = params.max_speed * 1.5;
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
            eaten += bite(here, bite_size(k));
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
            eaten += bite(gnaw, bite_size(k));
        }
    }

    // Hazards kill on contact.
    var died = 0u;
    if (params.hazard_active != 0u && distance(np, params.hazard_pos) < params.hazard_radius) {
        a.flags &= ~ALIVE;
        died = 1u;
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
