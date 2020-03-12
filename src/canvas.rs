use crate::geometry_utilities::types::*;
use lyon::math::*;
use winit::dpi::PhysicalSize;

pub struct CanvasView {
    pub zoom: f32,
    pub scroll: CanvasVector,
    pub resolution: PhysicalSize<u32>,
}

impl CanvasView {
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
