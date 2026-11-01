//! Renderer. Per frame, in order:
//! 1. **ground**: kitchen tiles, then walls and food from the material grid
//! 2. **agents**: every critter as an instanced quad read straight from the
//!    swarm's storage buffer
//! 3. **overlay**: hazards and the Queen, above the swarm so they never get lost
//! 4. **text**: the HUD, from a bitmap font

pub mod font;

use bytemuck::{Pod, Zeroable};

use crate::gpu::Gpu;
use crate::sim::material::{MaterialGrid, wgsl_kind_table};

const SCENE_WGSL: &str = include_str!("scene.wgsl");
/// HUD glyphs drawable per frame.
pub const MAX_GLYPHS: usize = 1024;

/// Layout must match `Camera` in `scene.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct CameraUniform {
    pub center: [f32; 2],
    pub half_extent: [f32; 2],
    pub focal: [f32; 2],
    pub time: f32,
    pub agent_size: f32,
    pub world_min: [f32; 2],
    pub world_max: [f32; 2],
    pub gather: f32,
    pub pixel_world: f32,
    pub mat_dim: [u32; 2],
    pub mat_cell: f32,
    pub _pad0: f32,
    pub _pad1: [f32; 2],
}

/// Layout must match `Overlay` in `scene.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct OverlayUniform {
    pub queen_pos: [f32; 2],
    pub queen_heading: [f32; 2],
    pub queen_radius: f32,
    /// 0..1
    pub queen_health: f32,
    pub queen_flash: f32,
    /// 0 healthy, 1 last stand, 2 dead
    pub queen_state: f32,
    pub hazard_pos: [f32; 2],
    pub hazard_radius: f32,
    /// 0 none, 1 warning, 2 active
    pub hazard_state: f32,
    pub hazard_warn: f32,
    pub guard_radius: f32,
    pub _pad: [f32; 2],
}

/// Layout must match `Hud` in `scene.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct HudUniform {
    viewport: [f32; 2],
    time: f32,
    _pad: f32,
}

/// One HUD character. Layout must match `Glyph` in `scene.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct GlyphInstance {
    /// Top-left in pixels.
    pub pos: [f32; 2],
    pub glyph: u32,
    /// Low byte: palette index; next byte: pixels per font pixel.
    pub style: u32,
}

/// Everything the renderer needs for one frame.
pub struct FrameData<'a> {
    pub camera: CameraUniform,
    pub overlay: OverlayUniform,
    pub glyphs: &'a [GlyphInstance],
    pub viewport: [u32; 2],
}

pub struct Renderer {
    format: wgpu::TextureFormat,
    agent_slots: u32,
    ground: wgpu::RenderPipeline,
    agents: wgpu::RenderPipeline,
    overlay: wgpu::RenderPipeline,
    text: wgpu::RenderPipeline,
    camera_buf: wgpu::Buffer,
    overlay_buf: wgpu::Buffer,
    hud_buf: wgpu::Buffer,
    glyph_buf: wgpu::Buffer,
    ground_bind: wgpu::BindGroup,
    agents_bind: wgpu::BindGroup,
    overlay_bind: wgpu::BindGroup,
    text_bind: wgpu::BindGroup,
}

impl Renderer {
    pub fn new(
        gpu: &Gpu,
        format: wgpu::TextureFormat,
        agents_buffer: &wgpu::Buffer,
        agent_slots: u32,
        material: &MaterialGrid,
    ) -> Self {
        let device = &gpu.device;
        let source = format!("{}\n{SCENE_WGSL}", wgsl_kind_table());
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scene"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let blend = Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING);
        let pipeline = |label: &str, vs: &str, fs: &str, blend: Option<wgpu::BlendState>| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: None,
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some(vs),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let ground = pipeline("ground", "vs_ground", "fs_ground", None);
        let agents = pipeline("agents", "vs_agent", "fs_agent", blend);
        let overlay = pipeline("overlay", "vs_ground", "fs_overlay", blend);
        let text = pipeline("text", "vs_text", "fs_text", blend);

        let uniform = |label: &str, size: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let camera_buf = uniform("camera", size_of::<CameraUniform>());
        let overlay_buf = uniform("overlay", size_of::<OverlayUniform>());
        let hud_buf = uniform("hud", size_of::<HudUniform>());
        let glyph_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("glyphs"),
            size: (MAX_GLYPHS * size_of::<GlyphInstance>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let font_data = font::packed();
        let font_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("font"),
            size: (font_data.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue
            .write_buffer(&font_buf, 0, bytemuck::cast_slice(&font_data));

        let bind = |label: &str, p: &wgpu::RenderPipeline, entries: &[(u32, &wgpu::Buffer)]| {
            let entries: Vec<_> = entries
                .iter()
                .map(|&(binding, b)| wgpu::BindGroupEntry {
                    binding,
                    resource: b.as_entire_binding(),
                })
                .collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &p.get_bind_group_layout(0),
                entries: &entries,
            })
        };
        let ground_bind = bind(
            "ground",
            &ground,
            &[
                (0, &camera_buf),
                (2, &material.kinds),
                (3, &material.amounts),
            ],
        );
        let agents_bind = bind("agents", &agents, &[(0, &camera_buf), (1, agents_buffer)]);
        let overlay_bind = bind("overlay", &overlay, &[(0, &camera_buf), (4, &overlay_buf)]);
        let text_bind = bind(
            "text",
            &text,
            &[(5, &hud_buf), (6, &glyph_buf), (7, &font_buf)],
        );
        Self {
            format,
            agent_slots,
            ground,
            agents,
            overlay,
            text,
            camera_buf,
            overlay_buf,
            hud_buf,
            glyph_buf,
            ground_bind,
            agents_bind,
            overlay_bind,
            text_bind,
        }
    }

    pub fn format(&self) -> wgpu::TextureFormat {
        self.format
    }

    /// Draws one frame into `target` and submits it.
    pub fn draw(&self, gpu: &Gpu, target: &wgpu::TextureView, frame: &FrameData) {
        let q = &gpu.queue;
        q.write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(&frame.camera));
        q.write_buffer(&self.overlay_buf, 0, bytemuck::bytes_of(&frame.overlay));
        let hud = HudUniform {
            viewport: [frame.viewport[0] as f32, frame.viewport[1] as f32],
            time: frame.camera.time,
            _pad: 0.0,
        };
        q.write_buffer(&self.hud_buf, 0, bytemuck::bytes_of(&hud));
        let glyphs = &frame.glyphs[..frame.glyphs.len().min(MAX_GLYPHS)];
        if !glyphs.is_empty() {
            q.write_buffer(&self.glyph_buf, 0, bytemuck::cast_slice(glyphs));
        }

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.ground);
            pass.set_bind_group(0, &self.ground_bind, &[]);
            pass.draw(0..3, 0..1);
            pass.set_pipeline(&self.agents);
            pass.set_bind_group(0, &self.agents_bind, &[]);
            pass.draw(0..6, 0..self.agent_slots);
            pass.set_pipeline(&self.overlay);
            pass.set_bind_group(0, &self.overlay_bind, &[]);
            pass.draw(0..3, 0..1);
            if !glyphs.is_empty() {
                pass.set_pipeline(&self.text);
                pass.set_bind_group(0, &self.text_bind, &[]);
                pass.draw(0..6, 0..glyphs.len() as u32);
            }
        }
        gpu.queue.submit([encoder.finish()]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_layouts_match_wgsl_sizes() {
        assert_eq!(size_of::<CameraUniform>(), 80);
        assert_eq!(size_of::<OverlayUniform>(), 64);
        assert_eq!(size_of::<HudUniform>(), 16);
        assert_eq!(size_of::<GlyphInstance>(), 16);
    }
}
