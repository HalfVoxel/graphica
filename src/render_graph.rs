use std::{rc::Rc};

use euclid::default::Size2D;
use lyon::math::{point, Rect};
use wgpu::{BindGroup, BlendState, Color, CommandEncoder, Device, Extent3d, LoadOp, TextureFormat, util::{StagingBelt}};

use crate::{blitter::{BlitGpuVertex, Blitter}, cache::ephermal_buffer_cache::{BufferRange, EphermalBufferCache}, cache::material_cache::{BindGroupEntryArc, BindingResourceArc, MaterialCache}, cache::render_pipeline_cache::{CachedRenderPipeline, RenderPipelineCache, RenderPipelineKey}, cache::render_texture_cache::RenderTextureCache, texture::{RenderTexture}};

struct Blit {
    source: GraphNode,
    target: GraphNode,
    source_rect: Rect,
    target_rect: Rect,
}

#[derive(Default)]
pub struct RenderGraph {
    nodes: Vec<RenderingPrimitive>,
}

enum RenderingPrimitive {
    UninitializedTexture(Size2D<u32>),
    Clear(GraphNode, Color),
    Blit(Blit),
}

pub struct GraphNode {
    index: usize,
}

#[derive(Debug)]
enum CompiledRenderingPrimitive {
    // Clear(RenderTexture, Color),
    Blit {
        source: RenderTexture,
        // target: RenderTexture,
        vbo: BufferRange,
        pipeline: CachedRenderPipeline,
        bind_group: Rc<BindGroup>,
    },
}

impl RenderGraph {
    fn push_primitive(&mut self, primitive: RenderingPrimitive) -> GraphNode {
        self.nodes.push(primitive);
        GraphNode {
            index: self.nodes.len() - 1,
        }
    }

    pub fn clear(&mut self, size: Size2D<u32>, color: Color) -> GraphNode {
        let tex = self.push_primitive(RenderingPrimitive::UninitializedTexture(size));
        self.push_primitive(RenderingPrimitive::Clear(tex, color))
    }

    // fn quad(&mut self, source: GraphNode, rect: Rect) -> GraphNode {

    // }

    pub fn blit(&mut self, source: GraphNode, target: GraphNode, source_rect: Rect, target_rect: Rect) -> GraphNode {
        self.push_primitive(RenderingPrimitive::Blit(Blit {
            source,
            target,
            source_rect,
            target_rect,
        }))
    }

    fn render(self, source: GraphNode) {}
}

pub struct RenderGraphCompiler<'a> {
    pub device: &'a Device,
    pub encoder: &'a mut CommandEncoder,
    pub blitter: &'a Blitter,
    pub render_pipeline_cache: &'a mut RenderPipelineCache,
    pub staging_belt: &'a mut StagingBelt,
    pub ephermal_buffer_cache: &'a mut EphermalBufferCache,
    pub render_texture_cache: &'a mut RenderTextureCache,
    pub material_cache: &'a mut MaterialCache,
}

struct RenderTextureSlot {
    texture: Option<RenderTexture>,
    size: Extent3d,
    format: TextureFormat,
    first_use_pass: Option<usize>,
    last_use_pass: Option<usize>,
}

#[derive(Debug)]
pub struct CompiledPass {
    target: RenderTexture,
    clear: Option<LoadOp<Color>>,
    ops: Vec<CompiledRenderingPrimitive>,
}

type PassIndex = usize;

#[derive(Debug)]
struct RenderTextureHandle(usize);

impl<'a> RenderGraphCompiler<'a> {

    fn blit_vertices(&mut self, source_uv_rect: &Rect, target_uv_rect: &Rect) -> BufferRange {
        puffin::profile_function!();
        let vertices = &[
            BlitGpuVertex {
                uv_source: point(source_uv_rect.min_x(), source_uv_rect.min_y()),
                uv_target: point(target_uv_rect.min_x(), target_uv_rect.min_y()),
            },
            BlitGpuVertex {
                uv_source: point(source_uv_rect.max_x(), source_uv_rect.min_y()),
                uv_target: point(target_uv_rect.max_x(), target_uv_rect.min_y()),
            },
            BlitGpuVertex {
                uv_source: point(source_uv_rect.max_x(), source_uv_rect.max_y()),
                uv_target: point(target_uv_rect.max_x(), target_uv_rect.max_y()),
            },
            BlitGpuVertex {
                uv_source: point(source_uv_rect.min_x(), source_uv_rect.max_y()),
                uv_target: point(target_uv_rect.min_x(), target_uv_rect.max_y()),
            },
        ];

        self.ephermal_buffer_cache.get(self.device, self.encoder, self.staging_belt, wgpu::BufferUsage::VERTEX, vertices)
    }

    pub fn compile(&mut self, render_graph: &RenderGraph, source: GraphNode, target_texture: &RenderTexture) -> Vec<CompiledPass> {
        puffin::profile_scope!("compile render graph");
        let mut sizes = vec![None; render_graph.nodes.len()];
        let mut usages = vec![0; render_graph.nodes.len()];

        fn trace(
            nodes: &[RenderingPrimitive],
            sizes: &mut [Option<Size2D<u32>>],
            usages: &mut [u32],
            node: &GraphNode,
        ) -> Size2D<u32> {
            if let Some(size) = sizes[node.index] {
                size
            } else {
                let size = match &nodes[node.index] {
                    RenderingPrimitive::UninitializedTexture(size) => *size,
                    RenderingPrimitive::Clear(target, _) => {
                        usages[node.index] += 1;
                        trace(nodes, sizes, usages, target)
                    }
                    RenderingPrimitive::Blit(Blit { source, target, .. }) => {
                        trace(nodes, sizes, usages, source);
                        trace(nodes, sizes, usages, target)
                    },
                };
                sizes[node.index] = Some(size);
                size
            }
        }

        {
            puffin::profile_scope!("trace");
            let output_size = trace(&render_graph.nodes, &mut sizes, &mut usages, &source);
            assert_eq!(output_size, Size2D::new(target_texture.size().width, target_texture.size().height));
        }

        let sizes = sizes
            .into_iter()
            .map(|s| s.unwrap_or_else(|| Size2D::new(0, 0)))
            .collect::<Vec<_>>();
        
        {
            puffin::profile_scope!("build");
            let mut passes = vec![];
            self.build(&render_graph.nodes, &sizes, &mut passes, &source, target_texture);
            passes
        }
    }

    fn pixel_to_uv_rect(pixel_rect: &Rect, texture_size: &Size2D<u32>) -> Rect {
        pixel_rect.scale(1.0/(texture_size.width as f32), 1.0/(texture_size.height as f32))
    }

    fn build(
        &mut self,
        nodes: &[RenderingPrimitive],
        sizes: &[Size2D<u32>],
        passes: &mut Vec<CompiledPass>,
        node: &GraphNode,
        target_texture: &RenderTexture,
    ) -> PassIndex {
        match &nodes[node.index] {
            RenderingPrimitive::UninitializedTexture(_) => {
                panic!();
                // Noop
                // passes.push(CompiledPass {
                //     clear: None,
                //     ops: vec![],
                // });
                // passes.len() - 1
            }
            RenderingPrimitive::Clear(_, color) => {
                passes.push(CompiledPass {
                    target: target_texture.clone(),
                    clear: Some(LoadOp::Clear(*color)),
                    ops: vec![],
                });
                passes.len() - 1
                // ops.push(CompiledRenderingPrimitive::Clear(target_texture, *color))
            }
            RenderingPrimitive::Blit(Blit {
                target,
                source,
                source_rect,
                target_rect,
            }) => {
                let source_texture_size = sizes[source.index];
                let target_texture_size = sizes[target.index];
                let target_texture_rect = Rect::from_size(target_texture_size.to_f32());
                if !target_texture_rect.intersects(target_rect) {
                    // Blit would end up outside the texture.
                    // We can ignore it.
                    self.build(nodes, sizes, passes, target, target_texture)
                } else {
                    let pipeline = self.render_pipeline_cache.get(self.device, RenderPipelineKey {
                        base: self.blitter.render_pipeline_base.clone().into(),
                        sample_count: target_texture.sample_count(),
                        depth_format: None,
                        target_format: target_texture.format(),
                        blend_state: BlendState::REPLACE,
                    }).to_owned();
                    let source_texture = self.render_texture_cache.temporary_render_texture(self.device, sizes[source.index], crate::config::TEXTURE_FORMAT);

                    let mat = self.material_cache.override_material(self.device, &self.blitter.material, &[BindGroupEntryArc {
                        binding: 1,
                        resource: BindingResourceArc::render_texture(Some(source_texture.clone())),
                    }]);
                    let op = CompiledRenderingPrimitive::Blit {
                        source: source_texture.clone(),
                        // target: target_texture,
                        pipeline,
                        bind_group: mat.bind_group().to_owned(),
                        // bind_group: self.blit_bind_group(source_texture.default_view().view),
                        vbo: self.blit_vertices(
                            &Self::pixel_to_uv_rect(source_rect, &source_texture_size),
                            &Self::pixel_to_uv_rect(target_rect, &target_texture_size),
                        ),
                    };

                    let source_pass = self.build(nodes, sizes, passes, source, &source_texture);

                    let pass = if target_rect.contains_rect(&target_texture_rect) {
                        // Blit covers the whole target, this means we can discard whatever was inside the target before.
                        passes.push(CompiledPass {
                            target: target_texture.clone(),
                            clear: None,
                            ops: vec![op],
                        });
                        passes.len() - 1
                    } else {
                        // Common path
                        // Blit does not cover the whole target
                        let target_pass = self.build(nodes, sizes, passes, target, &target_texture);
                        passes[target_pass].ops.push(op);
                        target_pass
                    };

                    // We are done using the source texture now
                    self.render_texture_cache.push(source_texture);

                    pass
                }
            }
        }
    }

    pub fn render(&mut self, passes: &[CompiledPass]) {
        puffin::profile_function!();
        for pass in passes {
            let color_attachment = wgpu::RenderPassColorAttachment {
                view: &pass.target.default_view().view,
                ops: wgpu::Operations {
                    load: pass.clear.unwrap_or(LoadOp::Load),
                    store: true,
                },
                resolve_target: None,
            };

            let mut render_pass = self.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("pass"),
                color_attachments: &[color_attachment],
                depth_stencil_attachment: None,
                // Some(wgpu::RenderPassDepthStencilAttachment {
                //     view: self.depth_texture_view.view,
                //     depth_ops: Some(wgpu::Operations {
                //         load: wgpu::LoadOp::Clear(0.0),
                //         store: true,
                //     }),
                //     stencil_ops: Some(wgpu::Operations {
                //         load: wgpu::LoadOp::Clear(0),
                //         store: true,
                //     }),
                // }),
            });

            for op in &pass.ops {
                match op {
                    CompiledRenderingPrimitive::Blit {
                        source,
                        vbo,
                        pipeline,
                        bind_group,
                    } => {
                        puffin::profile_scope!("blit");
                        render_pass.set_pipeline(&pipeline.pipeline);
                        render_pass.set_bind_group(0, &bind_group, &[]);
                        render_pass.set_index_buffer(self.blitter.ibo.slice(..), wgpu::IndexFormat::Uint32);
                        render_pass.set_vertex_buffer(0, vbo.as_slice());
                        render_pass.draw_indexed(0..6, 0, 0..1);
                    }
                }
            }

            {
                puffin::profile_scope!("drop render pass");
                drop(render_pass);
            }
        }
    }
}
