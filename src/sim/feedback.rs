//! Asynchronous GPU -> CPU feedback with no stalls.
//!
//! Each step the GPU reduces the swarm to a few hundred bytes: cumulative
//! counters plus per-workgroup partial sums. They are copied into one of a
//! small ring of staging buffers and mapped asynchronously; the CPU picks up
//! whichever result is newest, typically one or two frames old. Gameplay is
//! designed to tolerate that latency (see the design doc, section 6).
//!
//! Counters are cumulative, so a step whose readback is skipped (all slots
//! busy) loses nothing: the next summary includes it.

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use glam::{DVec2, Vec2};

use super::counter;

const FREE: u8 = 0;
const PENDING: u8 = 1;
const READY: u8 = 2;
const SLOTS: usize = 3;

/// Bytes of the counters block at the start of each readback.
pub const COUNTER_BYTES: u64 = counter::COUNT as u64 * 4;

/// What the CPU knows about the swarm without reading every agent.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SwarmSummary {
    /// Centre of the main body (critters near the focal point).
    pub centroid: Vec2,
    pub alive: u32,
    /// Critters in the main body.
    pub body: u32,
    /// Living critters within the guard radius of the Queen.
    pub near_queen: u32,
    /// Slots in use (alive or dead) below which agents are simulated.
    pub live_end: u32,
    /// Cumulative hatchlings placed.
    pub hatched_total: u32,
    /// Cumulative food eaten, fixed point (`material::FULL` = one cell).
    pub eaten_total: u32,
    /// Cumulative critters killed.
    pub died_total: u32,
    /// Step number this summary was taken at.
    pub step: u64,
}

impl SwarmSummary {
    /// Parses a readback: counters, then two vec4s per workgroup (see the
    /// reduce pass): (body x, body y, body count, alive), (near Queen, 0, 0, 0).
    pub fn parse(counters: &[u32], partials: &[[f32; 4]], step: u64) -> Self {
        let mut sum = DVec2::ZERO;
        let (mut body, mut alive, mut near) = (0.0f64, 0.0f64, 0.0f64);
        for [a, b] in partials.as_chunks::<2>().0 {
            sum += DVec2::new(a[0] as f64, a[1] as f64);
            body += a[2] as f64;
            alive += a[3] as f64;
            near += b[0] as f64;
        }
        let centroid = if body > 0.0 {
            (sum / body).as_vec2()
        } else {
            Vec2::ZERO
        };
        let c = |i: usize| counters.get(i).copied().unwrap_or(0);
        Self {
            centroid,
            alive: alive.round() as u32,
            body: body.round() as u32,
            near_queen: near.round() as u32,
            live_end: c(counter::LIVE_END),
            hatched_total: c(counter::HATCHED),
            eaten_total: c(counter::EATEN),
            died_total: c(counter::DIED),
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
        let size = COUNTER_BYTES + partials_bytes;
        let slots = (0..SLOTS)
            .map(|_| Slot {
                buffer: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("feedback staging"),
                    size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                state: Arc::new(AtomicU8::new(FREE)),
                step: 0,
            })
            .collect();
        Self {
            slots,
            size,
            latest: None,
        }
    }

    /// Records a copy of the counters and partials into a free slot, if any.
    /// Returns the slot index to pass to [`Feedback::after_submit`].
    pub fn record_copy(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        counters: &wgpu::Buffer,
        partials: &wgpu::Buffer,
        step: u64,
    ) -> Option<usize> {
        let index = self
            .slots
            .iter()
            .position(|s| s.state.load(Ordering::Acquire) == FREE)?;
        let slot = &mut self.slots[index];
        encoder.copy_buffer_to_buffer(counters, 0, &slot.buffer, 0, COUNTER_BYTES);
        encoder.copy_buffer_to_buffer(
            partials,
            0,
            &slot.buffer,
            COUNTER_BYTES,
            self.size - COUNTER_BYTES,
        );
        slot.state.store(PENDING, Ordering::Release);
        slot.step = step;
        Some(index)
    }

    /// Starts mapping a slot whose copy has just been submitted.
    pub fn after_submit(&mut self, index: usize) {
        let slot = &self.slots[index];
        let state = slot.state.clone();
        slot.buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                // On failure, free the slot so it is reused rather than leaked.
                let next = if result.is_ok() { READY } else { FREE };
                state.store(next, Ordering::Release);
            });
    }

    /// Collects any finished readbacks (non-blocking) and returns the newest.
    pub fn poll(&mut self, device: &wgpu::Device) -> Option<SwarmSummary> {
        let _ = device.poll(wgpu::PollType::Poll);
        for slot in &mut self.slots {
            if slot.state.load(Ordering::Acquire) != READY {
                continue;
            }
            let summary = {
                let view = slot.buffer.slice(..).get_mapped_range();
                view.ok().map(|v| {
                    let (head, tail) = v.split_at(COUNTER_BYTES as usize);
                    SwarmSummary::parse(
                        bytemuck::cast_slice::<u8, u32>(head),
                        bytemuck::cast_slice::<u8, [f32; 4]>(tail),
                        slot.step,
                    )
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
    fn summary_parses_counters_and_partials() {
        let partials = [
            [10.0, 20.0, 2.0, 3.0],
            [1.0, 0.0, 0.0, 0.0],
            [2.0, -8.0, 2.0, 2.0],
            [2.0, 0.0, 0.0, 0.0],
        ];
        let counters = [5, 3, 1500, 1, 0, 0, 0, 0];
        let s = SwarmSummary::parse(&counters, &partials, 7);
        assert_eq!((s.alive, s.body, s.near_queen), (5, 4, 3));
        assert_eq!(s.centroid, Vec2::new(3.0, 3.0));
        assert_eq!(
            (s.live_end, s.hatched_total, s.eaten_total, s.died_total),
            (5, 3, 1500, 1)
        );
        assert_eq!(s.step, 7);
        assert_eq!(SwarmSummary::parse(&[], &[], 0).alive, 0);
    }
}
