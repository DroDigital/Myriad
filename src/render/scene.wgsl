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
    mat_dim: vec2u,
    mat_cell: f32,
    pad0: f32,
    pad1: vec2f,
}

// Things drawn above the swarm: the Queen and hazards. Must match
// `OverlayUniform` in src/render/mod.rs.
struct Overlay {
    queen_pos: vec2f,
    queen_heading: vec2f,
    queen_radius: f32,
    queen_health: f32,
    queen_flash: f32,
    // 0 healthy, 1 last stand, 2 dead
    queen_state: f32,
    hazard_pos: vec2f,
    hazard_radius: f32,
    // 0 none, 1 warning, 2 active
    hazard_state: f32,
    hazard_warn: f32,
    guard_radius: f32,
    pad: vec2f,
}

struct Hud {
    viewport: vec2f,
    time: f32,
    pad: f32,
}

struct Glyph {
    pos: vec2f,
    glyph: u32,
    // low byte: palette index; next byte: pixels per font pixel
    style: u32,
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
@group(0) @binding(2) var<storage, read> mat_kind: array<u32>;
@group(0) @binding(3) var<storage, read> mat_amount: array<u32>;
@group(0) @binding(4) var<uniform> overlay: Overlay;
@group(0) @binding(5) var<uniform> hud: Hud;
@group(0) @binding(6) var<storage, read> glyphs: array<Glyph>;
@group(0) @binding(7) var<storage, read> font: array<u32>;

fn to_linear(c: vec3f) -> vec3f {
    return pow(c, vec3f(2.2));
}

fn hash2(p: vec2f) -> f32 {
    return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453);
}

// ---------------------------------------------------------------- material

fn kind_color(k: u32) -> vec3f {
    var table = KIND_COLOR;
    return table[min(k, KIND_COUNT - 1u)];
}

fn kind_flags(k: u32) -> u32 {
    var table = KIND_FLAGS;
    return table[min(k, KIND_COUNT - 1u)];
}

fn mat_index(c: vec2i) -> u32 {
    let cc = clamp(c, vec2i(0), vec2i(cam.mat_dim) - vec2i(1));
    return u32(cc.y) * cam.mat_dim.x + u32(cc.x);
}

// How much of kind `k` is at cell `c`, 0..1. Walls count as 1, and so does
// any solid food that still has something left: it blocks movement until it is
// completely gone, so it must stay visible until then.
fn coverage(c: vec2i, k: u32) -> f32 {
    let i = mat_index(c);
    if (mat_kind[i] != k) {
        return 0.0;
    }
    let flags = kind_flags(k);
    if ((flags & FLAG_BLOCKED) != 0u) {
        return 1.0;
    }
    let amount = mat_amount[i];
    if ((flags & FLAG_SOLID) != 0u) {
        return select(0.0, 1.0, amount > 0u);
    }
    return min(f32(amount) / f32(FULL), 1.0);
}

// Draws walls and food over the floor colour `base`. Food is a smooth blob
// (bilinear coverage, thresholded), so it shrinks visibly as it is eaten.
fn draw_material(w: vec2f, px: f32, base: vec3f) -> vec3f {
    let rel = (w - cam.world_min) / cam.mat_cell;
    let here = vec2i(floor(rel));
    let k = mat_kind[mat_index(here)];
    if (k == 0u) {
        return base;
    }
    let g = rel - vec2f(0.5);
    let b = vec2i(floor(g));
    let f = g - floor(g);
    let c00 = coverage(b, k);
    let c10 = coverage(b + vec2i(1, 0), k);
    let c01 = coverage(b + vec2i(0, 1), k);
    let c11 = coverage(b + vec2i(1, 1), k);
    let cov = mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
    let aa = clamp(px / cam.mat_cell, 0.02, 0.5);
    let alpha = smoothstep(0.45 - aa, 0.45 + aa, cov);
    let flags = kind_flags(k);
    var food = kind_color(k);
    if ((flags & FLAG_BLOCKED) != 0u) {
        // Cupboards and table legs: wood with a darker rim.
        let grain = 0.92 + 0.08 * sin(w.y * 3.0 + sin(w.x * 0.7) * 2.0);
        food *= grain * mix(0.7, 1.0, smoothstep(0.5, 0.95, cov));
    } else if ((flags & FLAG_SOLID) != 0u) {
        // Solid food keeps its shape until a cell is gone; gnawed cells darken.
        let fullness = min(f32(mat_amount[mat_index(here)]) / f32(FULL), 1.0);
        food *= mix(0.55, 1.05, fullness);
        let gx = (c10 - c00) + (c11 - c01);
        let gy = (c01 - c00) + (c11 - c10);
        food *= 1.0 + 0.12 * clamp(gx - gy, -1.0, 1.0);
    } else {
        // Thicker food reads brighter.
        food *= mix(0.72, 1.05, smoothstep(0.45, 1.0, cov));
        let gx = (c10 - c00) + (c11 - c01);
        let gy = (c01 - c00) + (c11 - c10);
        food *= 1.0 + 0.12 * clamp(gx - gy, -1.0, 1.0);
    }
    return mix(base, food, alpha);
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
        col = vec3f(0.30, 0.22, 0.16);
    } else {
        col = draw_material(w, px, col);
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

// ---------------------------------------------------------------- overlay
// Drawn above the swarm so the Queen and threats are never lost in it.

fn ant_sd(q: vec2f) -> f32 {
    let head = ellipse_sd(q, vec2f(0.0, 0.68), vec2f(0.17, 0.2));
    let thorax = ellipse_sd(q, vec2f(0.0, 0.18), vec2f(0.13, 0.22));
    let abdomen = ellipse_sd(q, vec2f(0.0, -0.45), vec2f(0.24, 0.42));
    var d = min(head, min(thorax, abdomen));
    let m = vec2f(abs(q.x), q.y);
    d = min(d, segment_sd(m, vec2f(0.05, 0.32), vec2f(0.5, 0.55)) - 0.03);
    d = min(d, segment_sd(m, vec2f(0.05, 0.18), vec2f(0.52, 0.12)) - 0.03);
    d = min(d, segment_sd(m, vec2f(0.05, 0.05), vec2f(0.48, -0.3)) - 0.03);
    return d;
}

fn over(dst: vec4f, col: vec3f, a: f32) -> vec4f {
    // Premultiplied "src over dst" where src = (col, a).
    return vec4f(col * a, a) + dst * (1.0 - a);
}

@fragment
fn fs_overlay(in: GroundOut) -> @location(0) vec4f {
    let w = cam.center + in.ndc * cam.half_extent;
    let px = cam.pixel_world;
    var out = vec4f(0.0);

    // Hazard: a soapy wet disk when active; a pulsing dashed ring as warning.
    if (overlay.hazard_state > 0.5) {
        let d = distance(w, overlay.hazard_pos);
        let r = overlay.hazard_radius;
        if (overlay.hazard_state > 1.5) {
            let wobble = 0.25 * sin(atan2(w.y - overlay.hazard_pos.y, w.x - overlay.hazard_pos.x) * 9.0 + cam.time * 4.0);
            let edge = r + wobble;
            let fill = 1.0 - smoothstep(edge - px, edge + px, d);
            let rim = 1.0 - smoothstep(0.0, 0.12 + px, abs(d - edge));
            let suds = 0.5 + 0.5 * sin(w.x * 2.3 + cam.time) * sin(w.y * 2.1 - cam.time * 1.3);
            out = over(out, linear(mix(vec3f(0.55, 0.70, 0.85), vec3f(0.92, 0.96, 1.0), suds)), fill * 0.55);
            out = over(out, linear(vec3f(0.20, 0.32, 0.50)), rim * 0.9);
        } else {
            let ang = atan2(w.y - overlay.hazard_pos.y, w.x - overlay.hazard_pos.x);
            let dash = step(0.0, sin(ang * 12.0 + cam.time * 3.0));
            let ring = 1.0 - smoothstep(0.0, 0.1 + px, abs(d - r));
            let pulse = 0.5 + 0.5 * sin(cam.time * 8.0);
            out = over(out, linear(vec3f(0.95, 0.55, 0.15)), ring * dash * (0.4 + 0.6 * pulse) * overlay.hazard_warn);
        }
    }

    // Queen: a gold beacon ring with a health arc, then her body on top.
    let to_q = w - overlay.queen_pos;
    let dq = length(to_q);
    let qr = overlay.queen_radius;
    if (overlay.queen_state < 1.5) {
        let beacon_r = qr * 1.9 + 0.15 * sin(cam.time * 3.0);
        let ring = 1.0 - smoothstep(0.0, 0.07 + px * 1.2, abs(dq - beacon_r));
        let ang = (atan2(to_q.x, to_q.y) / 6.2831853 + 0.5);
        let health_on = step(ang, overlay.queen_health);
        var ring_col = mix(vec3f(0.85, 0.15, 0.1), vec3f(1.0, 0.78, 0.25), health_on);
        if (overlay.queen_state > 0.5) {
            ring_col = vec3f(1.0, 0.15, 0.1) * (0.6 + 0.4 * sin(cam.time * 14.0));
        }
        out = over(out, linear(ring_col), ring * mix(0.35, 0.95, health_on));
    }
    // Body: the ant silhouette scaled up, facing her heading.
    let fwd = normalize(overlay.queen_heading);
    let right = vec2f(fwd.y, -fwd.x);
    let size = qr * 1.15;
    let q = vec2f(dot(to_q, right), dot(to_q, fwd)) / size;
    let d = ant_sd(q) * size;
    let outline = 1.0 - smoothstep(0.06, 0.06 + px * 1.5, d);
    let body = 1.0 - smoothstep(-px, px, d);
    var body_col = vec3f(0.95, 0.68, 0.18);
    let sheen = 1.0 - smoothstep(0.0, 0.25, length(q - vec2f(-0.08, -0.35)));
    body_col += vec3f(0.3, 0.25, 0.1) * sheen;
    body_col = mix(body_col, vec3f(1.0, 0.25, 0.2), overlay.queen_flash * 0.7);
    if (overlay.queen_state > 1.5) {
        body_col = vec3f(0.35, 0.3, 0.28);
    }
    out = over(out, linear(vec3f(0.12, 0.06, 0.02)), outline);
    out = over(out, linear(body_col), body);
    return out;
}

fn linear(c: vec3f) -> vec3f {
    return to_linear(c);
}

// ---------------------------------------------------------------- text

struct TextOut {
    @builtin(position) pos: vec4f,
    // Font-pixel coordinates within the glyph cell (outline border included).
    @location(0) fp: vec2f,
    @location(1) @interpolate(flat) glyph: u32,
    @location(2) @interpolate(flat) color: u32,
}

@vertex
fn vs_text(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> TextOut {
    var corners = array<vec2f, 6>(
        vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0),
        vec2f(0.0, 0.0), vec2f(1.0, 1.0), vec2f(0.0, 1.0),
    );
    let g = glyphs[ii];
    let scale = f32((g.style >> 8u) & 255u);
    // 5x7 glyph plus a 1-pixel outline on every side.
    let cell = vec2f(7.0, 9.0);
    let c = corners[vi];
    let px = g.pos + (c * cell - vec2f(1.0)) * scale;
    var o: TextOut;
    o.pos = vec4f(px.x / hud.viewport.x * 2.0 - 1.0, 1.0 - px.y / hud.viewport.y * 2.0, 0.0, 1.0);
    o.fp = c * cell - vec2f(1.0);
    o.glyph = g.glyph;
    o.color = g.style & 255u;
    return o;
}

fn font_bit(glyph: u32, x: i32, y: i32) -> bool {
    if (x < 0 || y < 0 || x > 4 || y > 6) {
        return false;
    }
    let row = font[glyph * 7u + u32(y)];
    return ((row >> u32(4 - x)) & 1u) == 1u;
}

@fragment
fn fs_text(in: TextOut) -> @location(0) vec4f {
    let x = i32(floor(in.fp.x));
    let y = i32(floor(in.fp.y));
    if (font_bit(in.glyph, x, y)) {
        var palette = array<vec3f, 5>(
            vec3f(1.0, 0.98, 0.94), vec3f(1.0, 0.75, 0.25), vec3f(1.0, 0.3, 0.22),
            vec3f(0.85, 0.82, 0.76), vec3f(0.55, 0.95, 0.45),
        );
        return vec4f(to_linear(palette[min(in.color, 4u)]), 1.0);
    }
    // Dark outline wherever a neighbouring pixel is lit, for legibility.
    var near = false;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            near = near || font_bit(in.glyph, x + dx, y + dy);
        }
    }
    if (near) {
        return vec4f(0.0, 0.0, 0.0, 0.75);
    }
    discard;
    return vec4f(0.0);
}
