//! Renderer: the ground plane, then every critter as an instanced quad that
//! reads its position straight from the swarm's storage buffer.

use bytemuck::{Pod, Zeroable};

use crate::gpu::Gpu;

const SCENE_WGSL: &str = include_str!("scene.wgsl");

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
    pub _pad: [f32; 2],
}

pub struct Renderer {
    format: wgpu::TextureFormat,
    ground: wgpu::RenderPipeline,
    agents: wgpu::RenderPipeline,
    camera: wgpu::Buffer,
    ground_bind: wgpu::BindGroup,
    agents_bind: wgpu::BindGroup,
}

impl Renderer {
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        agents_buffer: &wgpu::Buffer,
    ) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scene"),
            source: wgpu::ShaderSource::Wgsl(SCENE_WGSL.into()),
        });
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
        let agents = pipeline(
            "agents",
            "vs_agent",
            "fs_agent",
            Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
        );

        let camera = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera"),
            size: size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ground_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ground"),
            layout: &ground.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera.as_entire_binding(),
            }],
        });
        let agents_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("agents"),
            layout: &agents.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: agents_buffer.as_entire_binding(),
                },
            ],
        });
        Self {
            format,
            ground,
            agents,
            camera,
            ground_bind,
            agents_bind,
        }
    }

    pub fn format(&self) -> wgpu::TextureFormat {
        self.format
    }

    /// Draws one frame into `target` and submits it.
    pub fn draw(
        &self,
        gpu: &Gpu,
        target: &wgpu::TextureView,
        camera: &CameraUniform,
        agent_count: u32,
    ) {
        gpu.queue
            .write_buffer(&self.camera, 0, bytemuck::bytes_of(camera));
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
            if agent_count > 0 {
                pass.set_pipeline(&self.agents);
                pass.set_bind_group(0, &self.agents_bind, &[]);
                pass.draw(0..6, 0..agent_count);
            }
        }
        gpu.queue.submit([encoder.finish()]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_uniform_matches_wgsl_size() {
        assert_eq!(size_of::<CameraUniform>(), 64);
    }
}
