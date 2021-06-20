use egui::{ClippedMesh, CtxRef};
use wgpu::Device;
use winit::dpi::PhysicalSize;

pub struct EguiWrapper {
    pub platform: egui_winit_platform::Platform,
    egui_render_pass: egui_wgpu_backend::RenderPass,
}

impl EguiWrapper {
    pub fn new(device: &Device, size: PhysicalSize<u32>, scale_factor: f64) -> Self {
        // We use the egui_winit_platform crate as the platform.
        let platform = egui_winit_platform::Platform::new(egui_winit_platform::PlatformDescriptor {
            physical_width: size.width,
            physical_height: size.height,
            scale_factor,
            font_definitions: egui::FontDefinitions::default(),
            style: Default::default(),
        });

        // We use the egui_wgpu_backend crate as the render backend.
        let egui_render_pass = egui_wgpu_backend::RenderPass::new(&device, crate::config::TEXTURE_FORMAT);

        Self {
            platform,
            egui_render_pass,
        }
    }

    pub fn frame(&mut self, f: impl FnOnce(CtxRef)) -> Vec<ClippedMesh> {
        puffin::profile_function!();
        self.platform.begin_frame();
        f(self.platform.context());

        // End the UI frame. We could now handle the output and draw the UI with the backend.
        let (_output, paint_commands) = self.platform.end_frame();

        self.platform.context().tessellate(paint_commands)
    }

    pub fn render(
        &mut self,
        paint_jobs: &[ClippedMesh],
        device: &Device,
        view: &wgpu::TextureView,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        screen_descriptor: &egui_wgpu_backend::ScreenDescriptor,
    ) {
        puffin::profile_function!();
        self.egui_render_pass
            .update_texture(&device, &queue, &self.platform.context().texture());
        self.egui_render_pass.update_user_textures(&device, &queue);
        self.egui_render_pass
            .update_buffers(&device, &queue, &paint_jobs, &screen_descriptor);

        // Record all render passes.
        self.egui_render_pass
            .execute(encoder, view, paint_jobs, screen_descriptor, None);
    }
}
