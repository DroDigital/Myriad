// Swarm passes. Each `//! pass:<name>` section is compiled as its own module
// (prefixed with common.wgsl) so every pass gets an exact auto-derived layout.
// Order per step: count -> scan -> scatter -> steer.

//! pass:count
// Bucket each living agent into its grid cell and reserve a slot in that cell.
@group(0) @binding(1) var<storage, read> agents: array<Agent>;
@group(0) @binding(2) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> agent_cell: array<u32>;
@group(0) @binding(4) var<storage, read_write> agent_local: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.agent_count) {
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

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.agent_count) {
        return;
    }
    let c = agent_cell[i];
    if (c == INVALID) {
        return;
    }
    sorted[starts[c] + agent_local[i]] = i;
}

//! pass:steer
// Flocking + crowd pressure + seek toward the focal point, then integrate.
// Reads `agents`, writes `agents_next` (copied back after the pass).
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

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.agent_count) {
        return;
    }
    var a = agents[i];
    if ((a.flags & ALIVE) == 0u) {
        agents_next[i] = a;
        return;
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
    a.pos += a.vel * params.dt;
    a.crowding = crowding;

    // Walls: clamp and bleed off velocity into them.
    if (a.pos.x < params.world_min.x) {
        a.pos.x = params.world_min.x;
        a.vel.x = abs(a.vel.x) * 0.3;
    }
    if (a.pos.x > params.world_max.x) {
        a.pos.x = params.world_max.x;
        a.vel.x = -abs(a.vel.x) * 0.3;
    }
    if (a.pos.y < params.world_min.y) {
        a.pos.y = params.world_min.y;
        a.vel.y = abs(a.vel.y) * 0.3;
    }
    if (a.pos.y > params.world_max.y) {
        a.pos.y = params.world_max.y;
        a.vel.y = -abs(a.vel.y) * 0.3;
    }
    agents_next[i] = a;
}

//! pass:reduce
// Per-workgroup sums of (position, alive count) for the CPU to read back
// asynchronously: the swarm's centre without stalling the GPU.
@group(0) @binding(1) var<storage, read> agents: array<Agent>;
@group(0) @binding(2) var<storage, read_write> partials: array<vec4f>;

var<workgroup> acc: array<vec4f, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(local_invocation_index) t: u32,
    @builtin(workgroup_id) wg: vec3u,
) {
    var v = vec4f(0.0);
    if (gid.x < params.agent_count) {
        let a = agents[gid.x];
        if ((a.flags & ALIVE) != 0u) {
            v = vec4f(a.pos, 1.0, 0.0);
        }
    }
    acc[t] = v;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (t < s) {
            acc[t] += acc[t + s];
        }
        workgroupBarrier();
    }
    if (t == 0u) {
        partials[wg.x] = acc[0];
    }
}
