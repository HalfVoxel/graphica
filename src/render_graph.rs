use std::{collections::{hash_map::DefaultHasher, HashMap}, num::NonZeroU64, ops::Range, rc::Rc, sync::Arc};

use euclid::default::Size2D;
use lyon::math::{point, Rect};
use wgpu::{BindGroup, BlendState, Buffer, BufferAddress, BufferDescriptor, BufferUsage, Color, CommandEncoder, Device, Extent3d, LoadOp, RenderPipeline, TextureFormat, TextureView, util::{DeviceExt, StagingBelt}};

use crate::{blitter::{BlitGpuVertex, Blitter}, encoder::Encoder, material_cache::{BindGroupEntryArc, BindingResourceArc, MaterialCache}, render_pipeline_cache::{CachedRenderPipeline, RenderPipelineCache, RenderPipelineKey}, texture::{RenderTexture, Texture}, wgpu_utils::{as_u8_slice, create_buffer}};

#[derive(Debug)]
pub struct BufferRange {
    buffer: Arc<Buffer>,
    range: Range<BufferAddress>,
}

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

#[derive(Default)]
pub struct EphermalBufferCache {
    chunks: HashMap<BufferUsage, Vec<Chunk>>,
}

struct Chunk {
    buffer: Arc<Buffer>,
    used: u64,
    size: u64,
}

impl EphermalBufferCache {
    pub fn get<T>(&mut self, device: &Device, encoder: &mut CommandEncoder, staging_belt: &mut StagingBelt, mut usage: BufferUsage, contents: &[T]) -> BufferRange {
        usage |= BufferUsage::COPY_DST;

        let bytes = as_u8_slice(contents);
        let content_size = bytes.len() as u64;

        let chunks = self.chunks.entry(usage).or_default();
        let chunk_idx = if let Some(idx) = chunks.iter().position(|chunk| chunk.used + content_size <= chunk.size) {
            idx
        } else {
            let new_size = (content_size).next_power_of_two().max(chunks.last().map(|x| x.size).unwrap_or(0));
            println!("Creating a new buffer with size={}", new_size);
            chunks.push(Chunk {
                buffer: Arc::new(device.create_buffer(&BufferDescriptor {
                    label: Some("ephermal buffer"),
                    size: new_size,
                    usage,
                    mapped_at_creation: false,
                })),
                used: 0,
                size: new_size,
            });

            chunks.len() - 1
        };

        let chunk = &mut chunks[chunk_idx];
        if let Some(size) = NonZeroU64::new(content_size) {
            staging_belt.write_buffer(encoder, &chunk.buffer, chunk.used, size, device).copy_from_slice(bytes);
        }

        let result = BufferRange {
            buffer: chunk.buffer.clone(),
            range: chunk.used..chunk.used + content_size,
        };

        chunk.used += content_size;
        // Round up to the next multiple of 8
        // TODO: Investigate alignment requirements
        let remainder = chunk.used % 8;
        if remainder != 0 {
            chunk.used += 8 - remainder;
        }

        result
    }

    pub fn reset(&mut self) {
        for chunks in self.chunks.values_mut() {
            for chunk in chunks {
                chunk.used = 0;
            }
        }
    }
}

#[derive(Default)]
pub struct RenderTextureCache {
    render_textures: Vec<RenderTexture>,
}

impl RenderTextureCache {
    pub fn push(&mut self, rt: RenderTexture) {
        self.render_textures.push(rt);
    }

    pub fn temporary_render_texture(&mut self, device: &Device, size: Size2D<u32>, format: TextureFormat) -> RenderTexture {
        puffin::profile_function!();
        let best_tex = self.render_textures.iter().enumerate().filter(|(_,t)| {
            if t.format() != format {
                return false;
            }
            let tsize = t.size();
            tsize.width >= size.width && tsize.height >= size.height && tsize.width*tsize.height <= size.area()*4
        }).min_by_key(|(i,t)| t.size().width*t.size().height).map(|(i,t)| i);

        if let Some(index) = best_tex {
            self.render_textures.swap_remove(index)
        } else {
            let tex = Texture::new(device, wgpu::TextureDescriptor {
                label: Some("Temp texture"),
                size: Extent3d {
                    width: size.width,
                    height: size.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                // array_layer_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: crate::config::TEXTURE_FORMAT,
                usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::RENDER_ATTACHMENT,
            });
            let rt = RenderTexture::from(Rc::new(tex));
            rt
        }

        // self.render_textures.push(RenderTextureSlot {
        //     texture: None,
        //     size: Extent3d {
        //         width: size.width,
        //         height: size.height,
        //         depth_or_array_layers: 1,
        //     },
        //     format,
        //     first_use_pass: None,
        //     last_use_pass: None,
        // });
        // RenderTextureHandle(self.render_textures.len() - 1)
    
    }
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

    fn allocate_small_buffer() {}

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
        // let (vbo, _) = create_buffer(&self.device, vertices, wgpu::BufferUsage::VERTEX, None);
        // BufferRange {
        //     buffer: Arc::new(vbo),
        //     range: 0..4,
        // }
    }

    fn blit_bind_group(&mut self, source_texture: &TextureView) -> Rc<BindGroup> {
        puffin::profile_function!();
        Rc::new(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.blitter.bind_group_layout,
            label: Some("Blit bind group"),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Sampler(&self.blitter.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(source_texture),
                },
            ],
        }))
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
                        // bind_group: mat.bind_group().to_owned(),
                        bind_group: self.blit_bind_group(source_texture.default_view().view),
                        vbo: self.blit_vertices(
                            &source_rect.scale(1.0/source_texture_size.width as f32, 1.0/source_texture_size.height as f32),
                            &target_rect.scale(1.0/target_texture_size.width as f32, 1.0/target_texture_size.height as f32)
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
                    self.render_texture_cache.render_textures.push(source_texture);

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
                        render_pass.set_vertex_buffer(0, vbo.buffer.slice(vbo.range.clone()));
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
