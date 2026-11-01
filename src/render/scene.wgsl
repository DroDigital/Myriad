// Ground (kitchen tiles + focal marker) and instanced critters.
// `Camera` must match `CameraUniform` in src/render/mod.rs.

struct Camera {
    center: vec2f,
    half_extent: vec2f,
    focal: vec2f,
    time: f32,
    agent_size: f32,
    world_min: vec2f,
    world_max: vec2f,
    gather: f32,
    // World units covered by one pixel.
    pixel_world: f32,
    pad: vec2f,
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

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<storage, read> agents: array<Agent>;

fn to_linear(c: vec3f) -> vec3f {
    return pow(c, vec3f(2.2));
}

fn hash2(p: vec2f) -> f32 {
    return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453);
}

// ---------------------------------------------------------------- ground

struct GroundOut {
    @builtin(position) pos: vec4f,
    @location(0) ndc: vec2f,
}

@vertex
fn vs_ground(@builtin(vertex_index) vi: u32) -> GroundOut {
    let uv = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
    let ndc = uv * 2.0 - 1.0;
    var o: GroundOut;
    o.pos = vec4f(ndc, 0.0, 1.0);
    o.ndc = ndc;
    return o;
}

@fragment
fn fs_ground(in: GroundOut) -> @location(0) vec4f {
    let w = cam.center + in.ndc * cam.half_extent;
    let px = cam.pixel_world;

    // Kitchen floor tiles with grout.
    let tile = 4.0;
    let t = w / tile;
    let cell = floor(t);
    let f = fract(t);
    let edge = min(min(f.x, 1.0 - f.x), min(f.y, 1.0 - f.y)) * tile;
    let grout_w = 0.12;
    let grout = 1.0 - smoothstep(grout_w - px, grout_w + px, edge);
    let h = hash2(cell);
    var col = mix(vec3f(0.93, 0.89, 0.80), vec3f(0.86, 0.82, 0.73), h);
    col *= 0.97 + 0.03 * sin((w.x + w.y) * 0.15);
    col = mix(col, vec3f(0.62, 0.57, 0.50), grout);

    let inside = all(w >= cam.world_min) && all(w <= cam.world_max);
    if (!inside) {
        col = vec3f(0.36, 0.27, 0.20);
    }

    // Focal point: a pulsing amber ring that tightens on Gather.
    let d = length(w - cam.focal);
    let r = (1.2 - 0.4 * cam.gather) + 0.15 * sin(cam.time * 4.0);
    let ring = 1.0 - smoothstep(0.0, 0.08 + px * 1.5, abs(d - r));
    col = mix(col, vec3f(1.0, 0.62, 0.18), ring * 0.85);

    return vec4f(to_linear(col), 1.0);
}

// ---------------------------------------------------------------- critters

struct AgentOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
    @location(1) tint: vec3f,
    // Approximate uv units per pixel, for anti-aliasing and the far-LOD dot.
    @location(2) uv_per_px: f32,
}

@vertex
fn vs_agent(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> AgentOut {
    var corners = array<vec2f, 6>(
        vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
        vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0),
    );
    var o: AgentOut;
    let a = agents[ii];
    if ((a.flags & 1u) == 0u) {
        // Dead: emit a degenerate, clipped vertex.
        o.pos = vec4f(2.0, 2.0, 2.0, 1.0);
        return o;
    }
    let speed = length(a.vel);
    var fwd = vec2f(0.0, 1.0);
    if (speed > 1e-3) {
        fwd = a.vel / speed;
    }
    let right = vec2f(fwd.y, -fwd.x);
    let c = corners[vi];
    // Never smaller than ~1.5 px, so far critters still read as points.
    let s = max(cam.agent_size, cam.pixel_world * 1.5);
    let local = c * vec2f(s * 0.55, s);
    let world = a.pos + right * local.x + fwd * local.y;
    o.pos = vec4f((world - cam.center) / cam.half_extent, 0.0, 1.0);
    o.uv = c;
    o.uv_per_px = cam.pixel_world / s;
    o.tint = mix(vec3f(0.20, 0.11, 0.07), vec3f(0.36, 0.18, 0.08), a.seed);
    return o;
}

fn ellipse_sd(p: vec2f, c: vec2f, r: vec2f) -> f32 {
    let q = (p - c) / r;
    return (length(q) - 1.0) * min(r.x, r.y);
}

fn segment_sd(p: vec2f, a: vec2f, b: vec2f) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}

@fragment
fn fs_agent(in: AgentOut) -> @location(0) vec4f {
    // Back to isotropic units where 1.0 = body half-length.
    let q = vec2f(in.uv.x * 0.55, in.uv.y);
    let head = ellipse_sd(q, vec2f(0.0, 0.68), vec2f(0.17, 0.2));
    let thorax = ellipse_sd(q, vec2f(0.0, 0.18), vec2f(0.13, 0.22));
    let abdomen = ellipse_sd(q, vec2f(0.0, -0.45), vec2f(0.24, 0.42));
    var d = min(head, min(thorax, abdomen));
    // Three pairs of legs, mirrored.
    let m = vec2f(abs(q.x), q.y);
    d = min(d, segment_sd(m, vec2f(0.05, 0.32), vec2f(0.5, 0.55)) - 0.03);
    d = min(d, segment_sd(m, vec2f(0.05, 0.18), vec2f(0.52, 0.12)) - 0.03);
    d = min(d, segment_sd(m, vec2f(0.05, 0.05), vec2f(0.48, -0.3)) - 0.03);

    let aa = max(in.uv_per_px, 0.02);
    var alpha = 1.0 - smoothstep(-aa, aa, d);
    if (in.uv_per_px > 0.35) {
        // Far LOD: a soft dot instead of a silhouette.
        alpha = 1.0 - smoothstep(0.35, 1.0, length(q));
    }
    var col = in.tint;
    let sheen = 1.0 - smoothstep(0.0, 0.18, length(q - vec2f(-0.07, -0.35)));
    col += vec3f(0.35, 0.3, 0.25) * sheen * 0.6;
    return vec4f(to_linear(col) * alpha, alpha);
}
