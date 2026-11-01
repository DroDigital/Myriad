//! Top-down 2D camera that follows the swarm and pulls back as it grows.

use glam::Vec2;

#[derive(Clone, Copy, Debug)]
pub struct Camera2D {
    pub center: Vec2,
    /// Half the visible height in world units.
    pub half_height: f32,
    /// Viewport width / height.
    pub aspect: f32,
}

impl Camera2D {
    pub fn new(center: Vec2, half_height: f32, aspect: f32) -> Self {
        Self {
            center,
            half_height,
            aspect,
        }
    }

    pub fn half_extent(&self) -> Vec2 {
        Vec2::new(self.half_height * self.aspect, self.half_height)
    }

    /// Eases toward `target` and `half_height` (frame-rate independent).
    pub fn follow(&mut self, target: Vec2, half_height: f32, dt: f32) {
        let k_pos = 1.0 - (-dt * 3.0).exp();
        let k_zoom = 1.0 - (-dt * 1.5).exp();
        self.center = self.center.lerp(target, k_pos);
        self.half_height += (half_height - self.half_height) * k_zoom;
    }

    /// Converts a pixel position (origin top-left) to world space.
    pub fn screen_to_world(&self, px: Vec2, viewport: Vec2) -> Vec2 {
        let ndc = Vec2::new(px.x / viewport.x * 2.0 - 1.0, 1.0 - px.y / viewport.y * 2.0);
        self.center + ndc * self.half_extent()
    }

    /// World units covered by one pixel, given the viewport height in pixels.
    pub fn pixel_world(&self, viewport_height: u32) -> f32 {
        2.0 * self.half_height / viewport_height.max(1) as f32
    }
}

/// How much of the world to show for a swarm of `count` critters: close-up
/// macro at ten, wide at tens of thousands.
pub fn zoom_for_count(count: u32) -> f32 {
    (6.0 + (count as f32).sqrt() * 0.35).clamp(8.0, 120.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn screen_corners_map_to_view_edges() {
        let cam = Camera2D::new(Vec2::new(10.0, 5.0), 20.0, 2.0);
        let vp = Vec2::new(800.0, 400.0);
        let tl = cam.screen_to_world(Vec2::ZERO, vp);
        let br = cam.screen_to_world(vp, vp);
        assert!((tl - Vec2::new(-30.0, 25.0)).length() < 1e-4);
        assert!((br - Vec2::new(50.0, -15.0)).length() < 1e-4);
        assert!((cam.screen_to_world(vp * 0.5, vp) - cam.center).length() < 1e-4);
    }

    #[test]
    fn follow_converges() {
        let mut cam = Camera2D::new(Vec2::ZERO, 10.0, 1.0);
        for _ in 0..600 {
            cam.follow(Vec2::new(5.0, 5.0), 30.0, 1.0 / 60.0);
        }
        assert!((cam.center - Vec2::new(5.0, 5.0)).length() < 1e-3);
        assert!((cam.half_height - 30.0).abs() < 0.01);
    }

    #[test]
    fn zoom_grows_with_swarm_but_is_bounded() {
        assert_eq!(zoom_for_count(10), 8.0);
        assert!(zoom_for_count(20_000) > zoom_for_count(1_000));
        assert_eq!(zoom_for_count(u32::MAX), 120.0);
    }
}
