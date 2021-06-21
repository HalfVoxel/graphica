use std::{rc::Rc, sync::Arc};

use euclid::default::Size2D;
use lyon::math::{point, size, Rect};
use wgpu::{
    util::StagingBelt, BindGroup, BlendState, BufferUsage, Color, CommandEncoder, ComputePipeline, Device, Extent3d,
    LoadOp, TextureFormat, TextureUsage,
};
use wgpu_profiler::{wgpu_profiler, GpuProfiler};

use crate::{
    blitter::{BlitGpuVertex, Blitter},
    cache::ephermal_buffer_cache::{BufferRange, EphermalBufferCache},
    cache::material_cache::{BindGroupEntryArc, BindingResourceArc, MaterialCache},
    cache::render_texture_cache::RenderTextureCache,
    cache::{
        material_cache::Material,
        render_pipeline_cache::{CachedRenderPipeline, RenderPipelineBase, RenderPipelineCache, RenderPipelineKey},
    },
    geometry_utilities::types::{CanvasRect, UVRect},
    mipmap::Mipmapper,
    texture::RenderTexture,
    vertex::GPUVertex,
};

struct Blit {
    source: GraphNode,
    target: GraphNode,
    source_rect: CanvasRect,
    target_rect: CanvasRect,
}

#[derive(Default)]
pub struct RenderGraph {
    nodes: Vec<RenderingPrimitive>,
}

enum RenderingPrimitive {
    UninitializedTexture(Size2D<u32>),
    Clear(GraphNode, Color),
    Blit(Blit),
    Quad {
        target: GraphNode,
        rect: CanvasRect,
        pipeline: Arc<RenderPipelineBase>,
        material: Arc<Material>,
    },
    GenerateMipmaps {
        target: GraphNode,
    },
}

pub struct GraphNode {
    index: usize,
}

#[derive(Debug)]
pub enum CompiledComputePrimitive {
    GenerateMipmaps {
        target: RenderTexture,
        pipeline: Arc<ComputePipeline>,
        bind_groups: Vec<Rc<BindGroup>>,
    },
}

#[derive(Debug)]
pub enum CompiledRenderingPrimitive {
    // Clear(RenderTexture, Color),
    Blit {
        source: RenderTexture,
        // target: RenderTexture,
        vbo: BufferRange,
        pipeline: CachedRenderPipeline,
        bind_group: Rc<BindGroup>,
    },
    Render {
        vbo: BufferRange,
        ibo: BufferRange,
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

    pub fn quad(
        &mut self,
        target: GraphNode,
        rect: CanvasRect,
        pipeline: Arc<RenderPipelineBase>,
        material: Arc<Material>,
    ) -> GraphNode {
        self.push_primitive(RenderingPrimitive::Quad {
            target,
            rect,
            pipeline,
            material,
        })
    }

    pub fn blit(
        &mut self,
        source: GraphNode,
        target: GraphNode,
        source_rect: CanvasRect,
        target_rect: CanvasRect,
    ) -> GraphNode {
        self.push_primitive(RenderingPrimitive::Blit(Blit {
            source,
            target,
            source_rect,
            target_rect,
        }))
    }

    pub fn generate_mipmaps(&mut self, target: GraphNode) -> GraphNode {
        self.push_primitive(RenderingPrimitive::GenerateMipmaps { target })
    }

    fn render(self, _source: GraphNode) {}
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
    pub mipmapper: &'a Mipmapper,
    pub gpu_profiler: &'a mut GpuProfiler,
}

struct RenderTextureSlot {
    texture: Option<RenderTexture>,
    size: Extent3d,
    format: TextureFormat,
    first_use_pass: Option<usize>,
    last_use_pass: Option<usize>,
}

#[derive(Debug)]
pub enum CompiledPass {
    RenderPass {
        target: RenderTexture,
        clear: Option<LoadOp<Color>>,
        ops: Vec<CompiledRenderingPrimitive>,
    },
    ComputePass {
        ops: Vec<CompiledComputePrimitive>,
    },
}

type PassIndex = usize;

#[derive(Debug)]
struct RenderTextureHandle(usize);

#[derive(Debug, Clone)]
struct Usage {
    size: Size2D<u32>,
    usages: TextureUsage,
    requires_mipmaps: bool,
}

impl<'a> RenderGraphCompiler<'a> {
    fn blit_vertices(&mut self, source_uv_rect: &UVRect, target_uv_rect: &UVRect) -> BufferRange {
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

        self.ephermal_buffer_cache.get(
            self.device,
            self.encoder,
            self.staging_belt,
            wgpu::BufferUsage::VERTEX,
            vertices,
        )
    }

    pub fn compile(
        &mut self,
        render_graph: &RenderGraph,
        source: GraphNode,
        target_texture: &RenderTexture,
    ) -> Vec<CompiledPass> {
        puffin::profile_scope!("compile render graph");
        self.gpu_profiler.begin_scope("compile", self.encoder, self.device);
        let mut usages = vec![
            Usage {
                size: Size2D::new(0, 0),
                requires_mipmaps: false,
                usages: TextureUsage::empty(),
            };
            render_graph.nodes.len()
        ];

        let nodes = &render_graph.nodes;
        // Propagate backwards
        for i in (0..render_graph.nodes.len()).rev() {
            let mut usage = usages[i].clone();
            match &nodes[i] {
                RenderingPrimitive::UninitializedTexture(_) => {}
                RenderingPrimitive::Clear(_, _) => {
                    usage.usages |= TextureUsage::RENDER_ATTACHMENT;
                }
                RenderingPrimitive::Blit(_) => {
                    usage.usages |= TextureUsage::RENDER_ATTACHMENT;
                }
                RenderingPrimitive::Quad { .. } => {
                    usage.usages |= TextureUsage::RENDER_ATTACHMENT;
                }
                RenderingPrimitive::GenerateMipmaps { .. } => {
                    usage.usages |= TextureUsage::STORAGE;
                    usage.requires_mipmaps = true;
                }
            }

            match &nodes[i] {
                RenderingPrimitive::UninitializedTexture(_size) => {}
                RenderingPrimitive::Clear(target, _) => {
                    usages[target.index].requires_mipmaps |= usage.requires_mipmaps;
                    usages[target.index].usages |= usage.usages;
                }
                RenderingPrimitive::Blit(Blit { target, source, .. }) => {
                    usages[target.index].requires_mipmaps |= usage.requires_mipmaps;
                    usages[target.index].usages |= usage.usages;
                    usages[source.index].usages |= TextureUsage::SAMPLED;
                }
                RenderingPrimitive::Quad { target, .. } => {
                    usages[target.index].requires_mipmaps |= usage.requires_mipmaps;
                    usages[target.index].usages |= usage.usages;
                }
                RenderingPrimitive::GenerateMipmaps { target } => {
                    usages[target.index].requires_mipmaps |= usage.requires_mipmaps;
                    usages[target.index].usages |= usage.usages;
                }
            };

            usages[i] = usage;
        }

        // Propagate forwards
        for i in 0..render_graph.nodes.len() {
            let mut usage = usages[i].clone();
            match &nodes[i] {
                RenderingPrimitive::UninitializedTexture(size) => {
                    usage.size = *size;
                }
                RenderingPrimitive::Clear(target, _) => {
                    usage.size = usages[target.index].size;
                    usage.requires_mipmaps = usages[target.index].requires_mipmaps;
                    usage.usages = usages[target.index].usages;
                }
                RenderingPrimitive::Blit(Blit { source, target, .. }) => {
                    usage.size = usages[target.index].size;
                    usage.requires_mipmaps = usages[target.index].requires_mipmaps;
                    usage.usages = usages[target.index].usages;
                }
                RenderingPrimitive::Quad { target, .. } => {
                    usage.size = usages[target.index].size;
                    usage.requires_mipmaps = usages[target.index].requires_mipmaps;
                    usage.usages = usages[target.index].usages;
                }
                RenderingPrimitive::GenerateMipmaps { target } => {
                    usage.size = usages[target.index].size;
                    usage.requires_mipmaps = usages[target.index].requires_mipmaps;
                    usage.usages = usages[target.index].usages;
                }
            };
            usages[i] = usage;
        }

        // fn trace(
        //     nodes: &[RenderingPrimitive],
        //     sizes: &mut [Option<Size2D<u32>>],
        //     usages: &mut [Usage],
        //     node: &GraphNode,
        //     requires_mipmaps: bool,
        // ) -> Size2D<u32> {
        //     if let Some(size) = sizes[node.index] {
        //         size
        //     } else {
        //         let size = match &nodes[node.index] {
        //             RenderingPrimitive::UninitializedTexture(size) => *size,
        //             RenderingPrimitive::Clear(target, _) => {
        //                 trace(nodes, sizes, usages, target, requires_mipmaps)
        //             }
        //             RenderingPrimitive::Blit(Blit { source, target, .. }) => {
        //                 trace(nodes, sizes, usages, source, false);
        //                 trace(nodes, sizes, usages, target, requires_mipmaps)
        //             }
        //             RenderingPrimitive::Quad { target: source, .. } => trace(nodes, sizes, usages, source, requires_mipmaps),
        //             RenderingPrimitive::GenerateMipmaps { target} => trace(nodes, sizes, usages, target, true),
        //         };
        //         sizes[node.index] = Some(size);
        //         usages[node.index].requires_mipmaps = requires_mipmaps;
        //         size
        //     }
        // }

        {
            puffin::profile_scope!("trace");
            let output_size = usages[source.index].size;
            //trace(&render_graph.nodes, &mut sizes, &mut usages, &source, false);
            assert_eq!(
                output_size,
                Size2D::new(target_texture.size().width, target_texture.size().height)
            );
        }

        if !target_texture.usage().contains(usages[source.index].usages) {
            panic!(
                "The render target doesn't have sufficient usage flags. The render target has: {:?}, but required {:?}",
                target_texture.usage(),
                usages[source.index].usages
            );
        }

        // let sizes = sizes
        //     .into_iter()
        //     .map(|s| s.unwrap_or_else(|| Size2D::new(0, 0)))
        //     .collect::<Vec<_>>();

        let mut passes = vec![];
        {
            puffin::profile_scope!("build");
            self.build(&render_graph.nodes, &usages, &mut passes, &source, target_texture);
        }

        self.gpu_profiler.end_scope(self.encoder);
        passes
    }

    fn pixel_to_uv_rect(pixel_rect: &CanvasRect, texture_size: &Size2D<u32>) -> UVRect {
        pixel_rect
            .scale(1.0 / (texture_size.width as f32), 1.0 / (texture_size.height as f32))
            .cast_unit()
    }

    fn push_compute_op(
        passes: &mut Vec<CompiledPass>,
        target_pass: PassIndex,
        op: CompiledComputePrimitive,
    ) -> PassIndex {
        if let CompiledPass::ComputePass { ops, .. } = &mut passes[target_pass] {
            ops.push(op);
            target_pass
        } else {
            passes.push(CompiledPass::ComputePass { ops: vec![op] });
            passes.len() - 1
        }
    }

    fn push_render_op(
        passes: &mut Vec<CompiledPass>,
        target_texture: &RenderTexture,
        target_pass: PassIndex,
        op: CompiledRenderingPrimitive,
    ) -> PassIndex {
        if let CompiledPass::RenderPass { ops, .. } = &mut passes[target_pass] {
            ops.push(op);
            target_pass
        } else {
            passes.push(CompiledPass::RenderPass {
                target: target_texture.to_owned(),
                clear: Some(LoadOp::Load),
                ops: vec![op],
            });
            passes.len() - 1
        }
    }

    fn build(
        &mut self,
        nodes: &[RenderingPrimitive],
        usages: &[Usage],
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
                passes.push(CompiledPass::RenderPass {
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
                let source_texture_size = usages[source.index].size;
                let target_texture_size = usages[target.index].size;
                let target_texture_rect = CanvasRect::from_size(target_texture_size.to_f32().cast_unit());
                if !target_texture_rect.intersects(target_rect) {
                    // Blit would end up outside the texture.
                    // We can ignore it.
                    self.build(nodes, usages, passes, target, target_texture)
                } else {
                    let pipeline = self
                        .render_pipeline_cache
                        .get(
                            self.device,
                            RenderPipelineKey {
                                base: self.blitter.render_pipeline_base.clone().into(),
                                sample_count: target_texture.sample_count(),
                                depth_format: None,
                                target_format: target_texture.format(),
                                blend_state: BlendState::REPLACE,
                            },
                        )
                        .to_owned();
                    let source_texture = self.render_texture_cache.temporary_render_texture(
                        self.device,
                        usages[source.index].size,
                        crate::config::TEXTURE_FORMAT,
                        usages[source.index].requires_mipmaps,
                        usages[source.index].usages,
                    );

                    let mat = self.material_cache.override_material(
                        self.device,
                        &self.blitter.material,
                        &[BindGroupEntryArc {
                            binding: 1,
                            resource: BindingResourceArc::render_texture(Some(source_texture.clone())),
                        }],
                    );
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

                    let _source_pass = self.build(nodes, usages, passes, source, &source_texture);

                    let pass = if target_rect.contains_rect(&target_texture_rect) {
                        // Blit covers the whole target, this means we can discard whatever was inside the target before.
                        passes.push(CompiledPass::RenderPass {
                            target: target_texture.clone(),
                            clear: None,
                            ops: vec![op],
                        });
                        passes.len() - 1
                    } else {
                        // Common path
                        // Blit does not cover the whole target
                        let target_pass = self.build(nodes, usages, passes, target, target_texture);
                        Self::push_render_op(passes, target_texture, target_pass, op)
                    };

                    // We are done using the source texture now
                    self.render_texture_cache.push(source_texture);

                    pass
                }
            }
            RenderingPrimitive::Quad {
                target,
                rect,
                pipeline,
                material,
            } => {
                let target_pass = self.build(nodes, usages, passes, target, target_texture);
                let target_size = usages[target.index].size;

                let uv_rect = Self::pixel_to_uv_rect(rect, &target_size);
                let vbo = self.ephermal_buffer_cache.get(
                    self.device,
                    self.encoder,
                    self.staging_belt,
                    BufferUsage::VERTEX,
                    &[
                        point(uv_rect.min_x(), uv_rect.min_y()),
                        point(uv_rect.max_x(), uv_rect.min_y()),
                        point(uv_rect.max_x(), uv_rect.max_y()),
                        point(uv_rect.min_x(), uv_rect.max_y()),
                    ],
                );

                let ibo = self.ephermal_buffer_cache.get(
                    self.device,
                    self.encoder,
                    self.staging_belt,
                    BufferUsage::INDEX,
                    &[0, 1, 2, 3, 2, 0],
                );

                let op = CompiledRenderingPrimitive::Render {
                    vbo,
                    ibo,
                    pipeline: self
                        .render_pipeline_cache
                        .get(
                            self.device,
                            RenderPipelineKey {
                                base: pipeline.to_owned().into(),
                                sample_count: target_texture.sample_count(),
                                depth_format: None,
                                target_format: target_texture.format(),
                                blend_state: BlendState::REPLACE,
                            },
                        )
                        .to_owned(),
                    bind_group: material.bind_group().to_owned(),
                };

                Self::push_render_op(passes, target_texture, target_pass, op)
            }
            RenderingPrimitive::GenerateMipmaps { target } => {
                let target_pass = self.build(nodes, usages, passes, target, target_texture);

                let bind_groups = (1..target_texture.mip_level_count())
                    .map(|mip_level| {
                        puffin::profile_scope!("create bind group");
                        let mat = self.material_cache.override_material(
                            self.device,
                            &self.mipmapper.material,
                            &[
                                BindGroupEntryArc {
                                    binding: 0,
                                    resource: BindingResourceArc::Mipmap(Some((target_texture.clone(), mip_level - 1))),
                                },
                                BindGroupEntryArc {
                                    binding: 1,
                                    resource: BindingResourceArc::Mipmap(Some((target_texture.clone(), mip_level))),
                                },
                            ],
                        );

                        mat.bind_group().to_owned()

                        // let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        //     layout: &self.bind_group_layout,
                        //     entries: &[
                        //         wgpu::BindGroupEntry {
                        //             binding: 0,
                        //             resource: wgpu::BindingResource::TextureView(prev_level),
                        //         },
                        //         wgpu::BindGroupEntry {
                        //             binding: 1,
                        //             resource: wgpu::BindingResource::TextureView(current_level),
                        //         },
                        //     ],
                        //     label: None,
                        // });
                        // bind_groups.push(bind_group);
                        // prev_level = current_level;
                    })
                    .collect::<Vec<_>>();

                Self::push_compute_op(
                    passes,
                    target_pass,
                    CompiledComputePrimitive::GenerateMipmaps {
                        target: target_texture.to_owned(),
                        pipeline: self.mipmapper.pipeline.clone(),
                        bind_groups,
                    },
                )
            }
        }
    }

    pub fn render(&mut self, passes: &[CompiledPass]) {
        puffin::profile_function!();
        for pass in passes {
            match pass {
                CompiledPass::RenderPass { target, clear, ops } => {
                    let color_attachment = wgpu::RenderPassColorAttachment {
                        view: target.default_view().view,
                        ops: wgpu::Operations {
                            load: clear.unwrap_or(LoadOp::Load),
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

                    for op in ops {
                        match op {
                            CompiledRenderingPrimitive::Blit {
                                source: _,
                                vbo,
                                pipeline,
                                bind_group,
                            } => {
                                puffin::profile_scope!("op:blit");
                                wgpu_profiler!("op:blit", self.gpu_profiler, &mut render_pass, &self.device, {
                                    render_pass.set_pipeline(&pipeline.pipeline);
                                    render_pass.set_bind_group(0, bind_group, &[]);
                                    render_pass.set_index_buffer(self.blitter.ibo.slice(..), wgpu::IndexFormat::Uint32);
                                    render_pass.set_vertex_buffer(0, vbo.as_slice());
                                    render_pass.draw_indexed(0..6, 0, 0..1);
                                });
                            }
                            CompiledRenderingPrimitive::Render {
                                vbo,
                                ibo,
                                pipeline,
                                bind_group,
                            } => {
                                puffin::profile_scope!("op:render");
                                wgpu_profiler!("op:render", self.gpu_profiler, &mut render_pass, &self.device, {
                                    render_pass.set_pipeline(&pipeline.pipeline);
                                    render_pass.set_bind_group(0, bind_group, &[]);
                                    render_pass.set_index_buffer(ibo.as_slice(), wgpu::IndexFormat::Uint32);
                                    render_pass.set_vertex_buffer(0, vbo.as_slice());
                                    render_pass.draw_indexed(0..6, 0, 0..1);
                                });
                            }
                        }
                    }

                    {
                        puffin::profile_scope!("drop render pass");
                        drop(render_pass);
                    }
                }
                CompiledPass::ComputePass { ops } => {
                    let mut cpass = self
                        .encoder
                        .begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("cpass") });

                    for op in ops {
                        match op {
                            CompiledComputePrimitive::GenerateMipmaps {
                                target,
                                pipeline,
                                bind_groups,
                            } => {
                                wgpu_profiler!("op:generate_mipmaps", self.gpu_profiler, &mut cpass, &self.device, {
                                    assert!(target.mip_level_count() > 1);
                                    // assert!((texture.descriptor.size.width & (texture.descriptor.size.width - 1)) == 0, "Texture width must be a power of two. Found {}", texture.descriptor.size.width);
                                    // assert!((texture.descriptor.size.height & (texture.descriptor.size.height - 1)) == 0, "Texture height must be a power of two. Found {}", texture.descriptor.size.height);

                                    let mut width = target.size().width;
                                    let mut height = target.size().height;

                                    cpass.set_pipeline(pipeline);

                                    for bind_group in bind_groups {
                                        width = (width / 2).max(1);
                                        height = (height / 2).max(1);
                                        let local_size: u32 = 8;
                                        wgpu_profiler!("op:dispatch", self.gpu_profiler, &mut cpass, &self.device, {
                                            cpass.set_bind_group(0, bind_group, &[]);
                                            cpass.dispatch(
                                                (width + local_size - 1) / local_size,
                                                (height + local_size - 1) / local_size,
                                                1,
                                            );
                                        });
                                    }
                                });
                            }
                        }
                    }

                    {
                        puffin::profile_scope!("drop cpass");
                        drop(cpass);
                    }
                }
            }
        }
    }
}
