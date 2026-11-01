//! Windowless runs: scripted steering, timing, stats and an optional PNG.
//! Used for benchmarks, CI smoke tests and screenshots.

use std::fmt;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use glam::Vec2;

use crate::game::Controls;
use crate::gpu::Gpu;
use crate::session::{Session, SessionConfig};
use crate::sim::stats::SwarmStats;

#[derive(Clone, Debug)]
pub struct HeadlessOptions {
    pub agents: u32,
    pub capacity: u32,
    pub frames: u32,
    pub width: u32,
    pub height: u32,
    pub seed: u64,
    pub screenshot: Option<PathBuf>,
}

impl Default for HeadlessOptions {
    fn default() -> Self {
        Self {
            agents: 20_000,
            capacity: 32_768,
            frames: 600,
            width: 1280,
            height: 720,
            seed: 1,
            screenshot: None,
        }
    }
}

#[derive(Debug)]
pub struct HeadlessReport {
    pub adapter: String,
    pub agents: u32,
    pub frames: u32,
    pub ms_per_step: f64,
    pub stats: SwarmStats,
    pub screenshot: Option<PathBuf>,
}

impl fmt::Display for HeadlessReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "adapter: {}", self.adapter)?;
        writeln!(
            f,
            "simulated {} agents for {} steps: {:.3} ms/step",
            self.agents, self.frames, self.ms_per_step
        )?;
        write!(f, "swarm: {}", self.stats)?;
        if let Some(path) = &self.screenshot {
            write!(f, "\nscreenshot: {}", path.display())?;
        }
        Ok(())
    }
}

/// A figure-eight steering path, so headless runs exercise real flow.
pub fn scripted_controls(time: f32, center: Vec2) -> Controls {
    let target = center + Vec2::new(40.0 * (time * 0.25).sin(), 25.0 * (time * 0.5).sin());
    Controls {
        pointer: Some(target),
        gather: if (time * 0.2).sin() > 0.6 { 1.0 } else { 0.0 },
        ..Default::default()
    }
}

pub const CAPTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

pub fn run(opts: &HeadlessOptions) -> Result<HeadlessReport> {
    let gpu = Gpu::headless()?;
    log::info!("adapter: {}", gpu.adapter_summary());
    let cfg = SessionConfig {
        initial_agents: opts.agents,
        capacity: opts.capacity.max(opts.agents),
        seed: opts.seed,
        ..Default::default()
    };
    let aspect = opts.width as f32 / opts.height.max(1) as f32;
    let mut session = Session::new(&gpu, CAPTURE_FORMAT, aspect, &cfg);
    let center = (cfg.bounds.min + cfg.bounds.max) * 0.5;

    gpu.wait_idle();
    let start = Instant::now();
    for _ in 0..opts.frames {
        let controls = scripted_controls(session.game.time, center);
        session.step(&gpu, &controls);
        session
            .game
            .update_camera(session.swarm.count(), crate::session::SIM_DT);
    }
    gpu.wait_idle();
    let elapsed = start.elapsed().as_secs_f64();

    let agents = session.swarm.read_agents(&gpu)?;
    let stats = SwarmStats::compute(&agents, session.game.focal, session.swarm.bounds());

    let screenshot = match &opts.screenshot {
        Some(path) => {
            let image = capture(&gpu, &session, opts.width, opts.height)?;
            if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
                std::fs::create_dir_all(dir)?;
            }
            image
                .save(path)
                .with_context(|| format!("saving {}", path.display()))?;
            Some(path.clone())
        }
        None => None,
    };

    Ok(HeadlessReport {
        adapter: gpu.adapter_summary(),
        agents: session.swarm.count(),
        frames: opts.frames,
        ms_per_step: if opts.frames > 0 {
            elapsed * 1000.0 / opts.frames as f64
        } else {
            0.0
        },
        stats,
        screenshot,
    })
}

/// Renders the current frame offscreen and returns it as an image.
pub fn capture(gpu: &Gpu, session: &Session, width: u32, height: u32) -> Result<image::RgbaImage> {
    assert_eq!(session.renderer.format(), CAPTURE_FORMAT);
    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("capture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: CAPTURE_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    session.draw(gpu, &view, height);

    let row = width * 4;
    let padded_row =
        row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("capture staging"),
        size: padded_row as u64 * height as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("capture copy"),
        });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit([encoder.finish()]);
    let bytes: Vec<u8> = gpu.map_and_read(&staging)?;
    let mut pixels = Vec::with_capacity((row * height) as usize);
    for y in 0..height as usize {
        let start = y * padded_row as usize;
        pixels.extend_from_slice(&bytes[start..start + row as usize]);
    }
    image::RgbaImage::from_raw(width, height, pixels).context("capture buffer size mismatch")
}
