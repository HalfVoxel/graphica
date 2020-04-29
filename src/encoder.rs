use crate::blitter::Blitter;
use crate::main::Texture;
use std::sync::Arc;
use wgpu::{CommandEncoder, Device};

pub struct Encoder<'a> {
    pub device: &'a Device,
    pub encoder: &'a mut CommandEncoder,
    pub multisampled_render_target: Option<&'a wgpu::TextureView>,
    pub scratch_texture: Arc<Texture>,
    pub target_texture: &'a wgpu::TextureView,
    pub depth_texture_view: &'a wgpu::TextureView,
    pub blitter: &'a Blitter,
    pub resolution: wgpu::Extent3d,
}

impl Encoder<'_> {
    pub fn begin_msaa_render_pass<'a>(&'a mut self, clear: Option<wgpu::Color>) -> wgpu::RenderPass<'a> {
        // A resolve target is only supported if the attachment actually uses anti-aliasing
        // So if sample_count == 1 then we must render directly to the target texture
        let color_attachment = if let Some(msaa_target) = &self.multisampled_render_target {
            wgpu::RenderPassColorAttachmentDescriptor {
                attachment: msaa_target,
                load_op: if clear.is_some() {
                    wgpu::LoadOp::Clear
                } else {
                    wgpu::LoadOp::Load
                },
                store_op: wgpu::StoreOp::Store,
                clear_color: clear.unwrap_or(wgpu::Color::BLACK),
                resolve_target: Some(&self.target_texture),
            }
        } else {
            wgpu::RenderPassColorAttachmentDescriptor {
                attachment: &self.target_texture,
                load_op: wgpu::LoadOp::Clear,
                store_op: wgpu::StoreOp::Store,
                clear_color: wgpu::Color::WHITE,
                resolve_target: None,
            }
        };

        self.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[color_attachment],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachmentDescriptor {
                attachment: self.depth_texture_view,
                depth_load_op: wgpu::LoadOp::Clear,
                depth_store_op: wgpu::StoreOp::Store,
                stencil_load_op: wgpu::LoadOp::Clear,
                stencil_store_op: wgpu::StoreOp::Store,
                clear_depth: 0.0,
                clear_stencil: 0,
            }),
        })
    }

    pub fn begin_render_pass<'a>(&'a mut self, depth: bool) -> wgpu::RenderPass<'a> {
        let color_attachment = wgpu::RenderPassColorAttachmentDescriptor {
            attachment: &self.target_texture,
            load_op: wgpu::LoadOp::Load,
            store_op: wgpu::StoreOp::Store,
            clear_color: wgpu::Color::WHITE,
            resolve_target: None,
        };

        self.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[color_attachment],
            depth_stencil_attachment: if depth {
                Some(wgpu::RenderPassDepthStencilAttachmentDescriptor {
                    attachment: self.depth_texture_view,
                    depth_load_op: wgpu::LoadOp::Clear,
                    depth_store_op: wgpu::StoreOp::Store,
                    stencil_load_op: wgpu::LoadOp::Clear,
                    stencil_store_op: wgpu::StoreOp::Store,
                    clear_depth: 0.0,
                    clear_stencil: 0,
                })
            } else {
                None
            },
        })
    }
}
