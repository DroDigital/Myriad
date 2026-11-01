//! GPU device setup and small helpers shared by the simulation and renderer.

use anyhow::{Context, Result, anyhow};

/// The wgpu objects every system needs.
pub struct Gpu {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl Gpu {
    /// Creates a device with no window, for tests, benchmarks and screenshots.
    pub fn headless() -> Result<Self> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        pollster::block_on(Self::from_instance(instance, None))
    }

    /// Picks an adapter (compatible with `surface` when given) and opens a device.
    pub async fn from_instance(
        instance: wgpu::Instance,
        surface: Option<&wgpu::Surface<'_>>,
    ) -> Result<Self> {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: surface,
                ..Default::default()
            })
            .await
            .context("no suitable GPU adapter found")?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("myriad"),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            })
            .await
            .context("failed to open GPU device")?;
        Ok(Self {
            instance,
            adapter,
            device,
            queue,
        })
    }

    /// Short human-readable adapter description, e.g. for logs and the window title.
    pub fn adapter_summary(&self) -> String {
        let info = self.adapter.get_info();
        format!("{} ({:?}, {:?})", info.name, info.backend, info.device_type)
    }

    /// Blocks until all submitted GPU work has finished.
    pub fn wait_idle(&self) {
        if let Err(e) = self.device.poll(wgpu::PollType::wait_indefinitely()) {
            log::warn!("device poll failed: {e}");
        }
    }

    /// Copies the first `count` elements of `src` back to the CPU (blocking).
    ///
    /// `src` must have `COPY_SRC` usage. Meant for tests, stats and debugging,
    /// never for per-frame gameplay.
    pub fn read_buffer<T: bytemuck::Pod>(
        &self,
        src: &wgpu::Buffer,
        count: usize,
    ) -> Result<Vec<T>> {
        let size = (count * size_of::<T>()) as u64;
        if size == 0 {
            return Ok(Vec::new());
        }
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback"),
            });
        encoder.copy_buffer_to_buffer(src, 0, &staging, 0, size);
        self.queue.submit([encoder.finish()]);
        self.map_and_read(&staging)
    }

    /// Maps a `MAP_READ` buffer, waits for it, and copies its contents out.
    pub fn map_and_read<T: bytemuck::Pod>(&self, staging: &wgpu::Buffer) -> Result<Vec<T>> {
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.wait_idle();
        rx.recv()
            .context("map callback never ran")?
            .map_err(|e| anyhow!("buffer map failed: {e}"))?;
        let out = {
            let view = slice
                .get_mapped_range()
                .map_err(|e| anyhow!("mapped range: {e:?}"))?;
            bytemuck::cast_slice::<u8, T>(&view).to_vec()
        };
        staging.unmap();
        Ok(out)
    }
}
