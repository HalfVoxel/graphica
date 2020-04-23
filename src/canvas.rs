use crate::geometry_utilities::types::*;
use lyon::math::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use winit::dpi::PhysicalSize;

#[derive(Copy, Clone)]
pub struct CanvasView {
    pub zoom: f32,
    pub scroll: CanvasVector,
    pub resolution: PhysicalSize<u32>,
}

impl CanvasView {
    pub fn hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        hasher.write_u32(self.zoom.to_bits());
        hasher.write_u32(self.scroll.x.to_bits());
        hasher.write_u32(self.scroll.y.to_bits());
        self.resolution.hash(&mut hasher);
        hasher.finish()
    }

    pub fn visible_canvas_region(&self) -> euclid::Rect<f32, CanvasSpace> {
        let topleft = self.screen_to_canvas_point(point(0.0, 0.0));
        let bottomright =
            self.screen_to_canvas_point(point(self.resolution.width as f32, self.resolution.height as f32));
        rect(
            topleft.x,
            topleft.y,
            bottomright.x - topleft.x,
            bottomright.y - topleft.y,
        )
    }

    pub fn canvas_to_screen_rect(&self, r: euclid::Rect<f32, CanvasSpace>) -> euclid::Rect<f32, ScreenSpace> {
        let mn = self.canvas_to_screen_point(r.min());
        let mx = self.canvas_to_screen_point(r.max());
        rect(
            mn.x,
            mn.y,
            mx.x - mn.x,
            mx.y - mn.y,
        )
    }

    pub fn screen_to_normalized(&self, p: ScreenPoint) -> Point {
        point(
            p.x / self.resolution.width as f32,
            1.0 - p.y / self.resolution.height as f32,
        )
    }

    pub fn screen_to_canvas_scale(&self) -> euclid::Scale<f32, ScreenSpace, CanvasSpace> {
        euclid::Scale::new(1.0 / self.zoom)
    }

    pub fn canvas_to_screen_scale(&self) -> euclid::Scale<f32, CanvasSpace, ScreenSpace> {
        euclid::Scale::new(self.zoom)
    }

    pub fn screen_to_canvas_point(&self, point: ScreenPoint) -> CanvasPoint {
        (point - vector(self.resolution.width as f32, self.resolution.height as f32) * 0.5)
            * self.screen_to_canvas_scale()
            + self.scroll
    }

    pub fn screen_to_canvas_vector(&self, vector: ScreenVector) -> CanvasVector {
        vector * self.screen_to_canvas_scale()
    }

    pub fn canvas_to_screen_point(&self, point: CanvasPoint) -> ScreenPoint {
        (point - self.scroll) * self.canvas_to_screen_scale()
            + vector(self.resolution.width as f32, self.resolution.height as f32) * 0.5
    }
}
