use crate::blitter::Blitter;
use crate::texture::{RenderTextureView, Texture};
use std::sync::Arc;
use wgpu::{CommandEncoder, Device};

pub struct Encoder<'a> {
    pub device: &'a Device,
    pub encoder: &'a mut CommandEncoder,
    pub multisampled_render_target: Option<RenderTextureView<'a>>,
    pub scratch_texture: Arc<Texture>,
    pub target_texture: RenderTextureView<'a>,
    pub depth_texture_view: RenderTextureView<'a>,
    pub blitter: &'a Blitter,
    pub resolution: wgpu::Extent3d,
}

impl Encoder<'_> {
    pub fn begin_msaa_render_pass<'a>(
        &'a mut self,
        clear: Option<wgpu::Color>,
        label: Option<&'a str>,
    ) -> wgpu::RenderPass<'a> {
        // A resolve target is only supported if the attachment actually uses anti-aliasing
        // So if sample_count == 1 then we must render directly to the target texture
        let color_attachment = if let Some(msaa_target) = &self.multisampled_render_target {
            wgpu::RenderPassColorAttachment {
                view: msaa_target.view,
                ops: wgpu::Operations {
                    load: if let Some(color) = clear {
                        wgpu::LoadOp::Clear(color)
                    } else {
                        wgpu::LoadOp::Load
                    },
                    store: true,
                },
                resolve_target: Some(self.target_texture.view),
            }
        } else {
            wgpu::RenderPassColorAttachment {
                view: self.target_texture.view,
                ops: wgpu::Operations {
                    load: if let Some(color) = clear {
                        wgpu::LoadOp::Clear(color)
                    } else {
                        wgpu::LoadOp::Load
                    },
                    store: true,
                },
                resolve_target: None,
            }
        };

        self.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: label.or(Some("msaa render pass")),
            color_attachments: &[color_attachment],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: self.depth_texture_view.view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0.0),
                    store: true,
                }),
                stencil_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0),
                    store: true,
                }),
            }),
        })
    }

    pub fn begin_render_pass(&mut self, depth: bool) -> wgpu::RenderPass {
        let color_attachment = wgpu::RenderPassColorAttachment {
            view: self.target_texture.view,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: true,
            },
            resolve_target: None,
        };

        self.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[color_attachment],
            depth_stencil_attachment: if depth {
                Some(wgpu::RenderPassDepthStencilAttachment {
                    view: self.depth_texture_view.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: true,
                    }),
                    stencil_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0),
                        store: true,
                    }),
                })
            } else {
                None
            },
        })
    }
}
