//! Asynchronous GPU -> CPU feedback with no stalls.
//!
//! The GPU reduces the swarm to a few hundred bytes of per-workgroup partial
//! sums each step. Those are copied into one of a small ring of staging
//! buffers and mapped asynchronously; the CPU picks up whichever result is
//! newest, typically one or two frames old. Gameplay is designed to tolerate
//! that latency (see the design doc, section 6).

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use glam::{DVec2, Vec2};

const FREE: u8 = 0;
const PENDING: u8 = 1;
const READY: u8 = 2;
const SLOTS: usize = 3;

/// What the CPU knows about the swarm without reading every agent.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SwarmSummary {
    pub centroid: Vec2,
    pub alive: u32,
    /// Step number this summary was taken at.
    pub step: u64,
}

impl SwarmSummary {
    /// Sums per-workgroup partials of (x, y, alive, _).
    pub fn from_partials(partials: &[[f32; 4]], step: u64) -> Self {
        let mut sum = DVec2::ZERO;
        let mut alive = 0.0f64;
        for p in partials {
            sum += DVec2::new(p[0] as f64, p[1] as f64);
            alive += p[2] as f64;
        }
        let centroid = if alive > 0.0 {
            (sum / alive).as_vec2()
        } else {
            Vec2::ZERO
        };
        Self {
            centroid,
            alive: alive.round() as u32,
            step,
        }
    }
}

struct Slot {
    buffer: wgpu::Buffer,
    state: Arc<AtomicU8>,
    step: u64,
}

pub struct Feedback {
    slots: Vec<Slot>,
    size: u64,
    latest: Option<SwarmSummary>,
}

impl Feedback {
    pub fn new(device: &wgpu::Device, partials_bytes: u64) -> Self {
        let slots = (0..SLOTS)
            .map(|_| Slot {
                buffer: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("feedback staging"),
                    size: partials_bytes,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                state: Arc::new(AtomicU8::new(FREE)),
                step: 0,
            })
            .collect();
        Self {
            slots,
            size: partials_bytes,
            latest: None,
        }
    }

    /// Records a copy of `partials` into a free slot, if any. Returns the slot
    /// index to pass to [`Feedback::after_submit`].
    pub fn record_copy(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        partials: &wgpu::Buffer,
        bytes: u64,
        step: u64,
    ) -> Option<usize> {
        let index = self
            .slots
            .iter()
            .position(|s| s.state.load(Ordering::Acquire) == FREE)?;
        let slot = &mut self.slots[index];
        encoder.copy_buffer_to_buffer(partials, 0, &slot.buffer, 0, bytes.min(self.size));
        slot.state.store(PENDING, Ordering::Release);
        slot.step = step;
        Some(index)
    }

    /// Starts mapping a slot whose copy has just been submitted.
    pub fn after_submit(&mut self, index: usize, bytes: u64) {
        let slot = &self.slots[index];
        let state = slot.state.clone();
        slot.buffer
            .slice(..bytes.min(self.size))
            .map_async(wgpu::MapMode::Read, move |result| {
                // On failure, free the slot so it is reused rather than leaked.
                let next = if result.is_ok() { READY } else { FREE };
                state.store(next, Ordering::Release);
            });
    }

    /// Collects any finished readbacks (non-blocking) and returns the newest.
    pub fn poll(&mut self, device: &wgpu::Device, bytes: u64) -> Option<SwarmSummary> {
        let _ = device.poll(wgpu::PollType::Poll);
        for slot in &mut self.slots {
            if slot.state.load(Ordering::Acquire) != READY {
                continue;
            }
            let summary = {
                let view = slot.buffer.slice(..bytes.min(self.size)).get_mapped_range();
                view.ok().map(|v| {
                    SwarmSummary::from_partials(bytemuck::cast_slice::<u8, [f32; 4]>(&v), slot.step)
                })
            };
            slot.buffer.unmap();
            slot.state.store(FREE, Ordering::Release);
            if let Some(s) = summary
                && self.latest.is_none_or(|l| s.step > l.step)
            {
                self.latest = Some(s);
            }
        }
        self.latest
    }

    pub fn latest(&self) -> Option<SwarmSummary> {
        self.latest
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_from_partials() {
        let partials = [[10.0, 20.0, 2.0, 0.0], [2.0, -8.0, 2.0, 0.0], [0.0; 4]];
        let s = SwarmSummary::from_partials(&partials, 7);
        assert_eq!(s.alive, 4);
        assert_eq!(s.centroid, Vec2::new(3.0, 3.0));
        assert_eq!(s.step, 7);
        assert_eq!(SwarmSummary::from_partials(&[], 0).alive, 0);
    }
}
