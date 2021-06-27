use core::panic;
use std::{collections::VecDeque, rc::Rc, sync::Arc};

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

#[derive(Debug)]
struct Blit {
    source: GraphNode,
    target: GraphNode,
    source_rect: CanvasRect,
    target_rect: CanvasRect,
    blend: BlendState,
}

#[derive(Default)]
pub struct RenderGraph {
    nodes: Vec<RenderingPrimitive>,
}

#[derive(Debug)]
enum RenderingPrimitive {
    UninitializedTexture(Size2D<u32>),
    Clear(GraphNode, Color),
    Blit(Blit),
    Mesh {
        target: GraphNode,
        vbo: BufferRange,
        ibo: BufferRange,
        pipeline: Arc<RenderPipelineBase>,
        material: Arc<Material>,
    },
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

impl RenderingPrimitive {
    fn reads(&self) -> impl Iterator<Item = GraphNode> {
        match self {
            RenderingPrimitive::Blit(Blit { source, .. }) => vec![source.clone()].into_iter(),
            _ => vec![].into_iter(),
        }
    }

    fn target(&self) -> Option<GraphNode> {
        match self {
            RenderingPrimitive::UninitializedTexture(_) => None,
            RenderingPrimitive::Clear(target, _) => Some(target.clone()),
            RenderingPrimitive::Blit(Blit { target, .. }) => Some(target.clone()),
            RenderingPrimitive::Mesh { target, .. } => Some(target.clone()),
            RenderingPrimitive::Quad { target, .. } => Some(target.clone()),
            RenderingPrimitive::GenerateMipmaps { target } => Some(target.clone()),
        }
    }

    fn writes(&self) -> impl Iterator<Item = GraphNode> {
        self.target().into_iter()
    }
}

#[derive(Clone, PartialEq, Debug)]
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

    pub fn mesh(
        &mut self,
        target: GraphNode,
        vbo: BufferRange,
        ibo: BufferRange,
        pipeline: Arc<RenderPipelineBase>,
        material: Arc<Material>,
    ) -> GraphNode {
        self.push_primitive(RenderingPrimitive::Mesh {
            target,
            vbo,
            ibo,
            pipeline,
            material,
        })
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
            blend: BlendState::REPLACE,
        }))
    }

    pub fn blend(
        &mut self,
        source: GraphNode,
        target: GraphNode,
        source_rect: CanvasRect,
        target_rect: CanvasRect,
        blend: BlendState,
    ) -> GraphNode {
        self.push_primitive(RenderingPrimitive::Blit(Blit {
            source,
            target,
            source_rect,
            target_rect,
            blend,
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
    logical_render_target: usize,
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
                logical_render_target: usize::MAX,
            };
            render_graph.nodes.len()
        ];

        let nodes = &render_graph.nodes;
        let mut logical_render_target_index = 0;
        // Propagate backwards
        for i in (0..render_graph.nodes.len()).rev() {
            let mut usage = usages[i].clone();
            match &nodes[i] {
                RenderingPrimitive::UninitializedTexture(size) => {
                    usage.logical_render_target = logical_render_target_index;
                    logical_render_target_index += 1;
                    usage.size = *size;
                }
                RenderingPrimitive::Clear(_, _) => {
                    usage.usages |= TextureUsage::RENDER_ATTACHMENT;
                }
                RenderingPrimitive::Blit(_) => {
                    usage.usages |= TextureUsage::RENDER_ATTACHMENT;
                }
                RenderingPrimitive::Quad { .. } => {
                    usage.usages |= TextureUsage::RENDER_ATTACHMENT;
                }
                RenderingPrimitive::Mesh { .. } => {
                    usage.usages |= TextureUsage::RENDER_ATTACHMENT;
                }
                RenderingPrimitive::GenerateMipmaps { .. } => {
                    usage.usages |= TextureUsage::STORAGE;
                    usage.requires_mipmaps = true;
                }
            }

            if let Some(target) = nodes[i].target() {
                usages[target.index].requires_mipmaps |= usage.requires_mipmaps;
                usages[target.index].usages |= usage.usages;
            }

            for source in nodes[i].reads() {
                // Assume reads are sampled with texture samplers. TODO: Not necessarily true.
                usages[source.index].usages |= TextureUsage::SAMPLED;
            }

            usages[i] = usage;
        }

        // Propagate forwards
        for i in 0..render_graph.nodes.len() {
            let mut usage = usages[i].clone();

            if let Some(target) = nodes[i].target() {
                usage.size = usages[target.index].size;
                usage.requires_mipmaps = usages[target.index].requires_mipmaps;
                usage.usages = usages[target.index].usages;
                usage.logical_render_target = usages[target.index].logical_render_target;
            }
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

        let mut logical_to_physical_render_targets = vec![None; logical_render_target_index];
        logical_to_physical_render_targets[usages[source.index].logical_render_target] = Some(target_texture.clone());

        let (sorted, logical_to_physical_render_targets) = {
            puffin::profile_scope!("schedule");
            self.topological_sort(
                &render_graph.nodes,
                &usages,
                &source,
                logical_to_physical_render_targets,
            )
        };
        // println!("{:#?}", sorted.iter().map(|i| &nodes[i.index]).collect::<Vec<_>>());

        let mut passes = vec![];
        {
            puffin::profile_scope!("build");
            // self.build(&render_graph.nodes, &usages, &mut passes, &source, target_texture);
            self.build_top(
                &render_graph.nodes,
                &sorted,
                &usages,
                &mut passes,
                &logical_to_physical_render_targets,
            );
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

    fn push_render_op_top(
        passes: &mut Vec<CompiledPass>,
        target_texture: &RenderTexture,
        op: CompiledRenderingPrimitive,
    ) {
        if let Some(CompiledPass::RenderPass { target, ops, .. }) = passes.last_mut() {
            if target == target_texture {
                ops.push(op);
                return;
            }
        }

        passes.push(CompiledPass::RenderPass {
            target: target_texture.to_owned(),
            clear: Some(LoadOp::Load),
            ops: vec![op],
        });
    }

    fn push_compute_op_top(passes: &mut Vec<CompiledPass>, op: CompiledComputePrimitive) {
        if let Some(CompiledPass::ComputePass { ops, .. }) = passes.last_mut() {
            ops.push(op);
        } else {
            passes.push(CompiledPass::ComputePass { ops: vec![op] });
        }
    }

    fn topological_sort(
        &mut self,
        nodes: &[RenderingPrimitive],
        usages: &[Usage],
        start_node: &GraphNode,
        // Mapping from logical render targets to physical render targets
        // May contain duplicates due to aliasing.
        mut logical_to_physical_render_targets: Vec<Option<RenderTexture>>,
    ) -> (Vec<GraphNode>, Vec<RenderTexture>) {
        #[derive(Default, Clone)]
        struct TopSortNode {
            remaining_dependencies: u32,
            remaining_reads_from_self: Vec<GraphNode>,
            remaining_readwrite_from_self: Option<GraphNode>,
            /// Tie-breaking bias. Higher values should be executed first.
            sort_order: u32,
        }

        struct LogicalRenderTexture(usize);
        struct PhysicalRenderTexture(usize);

        let mut topgraph = vec![TopSortNode::default(); nodes.len()];
        let mut queued = vec![false; nodes.len()];
        let mut que = VecDeque::new();
        let mut sort_order_index = 0;
        // Refcounts for the logical render targets.
        // Predefined render textures start at a refcount of 1 so that they do not get deallocated
        let mut logical_render_target_refcount = logical_to_physical_render_targets
            .iter()
            .map(|x| x.is_some() as usize)
            .collect::<Vec<_>>();
        let mut logical_render_target_pressure = vec![0; logical_to_physical_render_targets.len()];

        que.push_front(start_node.clone());
        while let Some(node) = que.pop_front() {
            topgraph[node.index].sort_order = sort_order_index;
            sort_order_index += 1;

            for target in nodes[node.index].reads() {
                logical_render_target_refcount[usages[target.index].logical_render_target] += 1;

                topgraph[target.index].remaining_reads_from_self.push(node.clone());
                topgraph[node.index].remaining_dependencies += 1;
                if !queued[target.index] {
                    queued[target.index] = true;
                    que.push_back(target);
                }
            }
            for target in nodes[node.index].writes() {
                logical_render_target_refcount[usages[target.index].logical_render_target] += 1;

                assert!(topgraph[target.index].remaining_readwrite_from_self.is_none());
                topgraph[target.index].remaining_readwrite_from_self = Some(node.clone());
                topgraph[node.index].remaining_dependencies += 1;
                if !queued[target.index] {
                    queued[target.index] = true;
                    que.push_front(target);
                }
            }
        }

        let mut available_nodes = vec![];
        for (index, node) in topgraph.iter().enumerate() {
            if node.remaining_dependencies == 0 {
                available_nodes.push(GraphNode { index });
            }
        }

        fn reduce_rt_refcount(
            logical_render_target: usize,
            logical_render_target_refcount: &mut [usize],
            render_texture_cache: &mut RenderTextureCache,
            logical_to_physical_render_targets: &[Option<RenderTexture>],
        ) {
            assert!(
                logical_render_target_refcount[logical_render_target] > 0,
                "Refcount for texture went below zero"
            );
            logical_render_target_refcount[logical_render_target] -= 1;
            if logical_render_target_refcount[logical_render_target] == 0 {
                // Deallocate
                render_texture_cache.push(
                    logical_to_physical_render_targets[logical_render_target]
                        .clone()
                        .unwrap(),
                );
            }
        }

        let mut sorted = vec![];
        while !available_nodes.is_empty() {
            let best_node = available_nodes
                .iter()
                .cloned()
                .enumerate()
                .filter(|(_, node)| {
                    // Check if we have an allocation for this
                    match &nodes[node.index] {
                        RenderingPrimitive::UninitializedTexture(size) => self.render_texture_cache.has(
                            *size,
                            crate::config::TEXTURE_FORMAT,
                            usages[node.index].requires_mipmaps,
                            usages[node.index].usages,
                        ),
                        // Render target already in-use
                        _ => true,
                    }
                })
                .max_by_key(|(_, node)| topgraph[node.index].sort_order)
                .or_else(|| {
                    available_nodes.iter().cloned().enumerate().max_by_key(|(_, node)| {
                        (
                            logical_render_target_pressure[usages[node.index].logical_render_target],
                            topgraph[node.index].sort_order,
                        )
                    })
                });

            if let Some((idx, best_node)) = best_node {
                available_nodes.swap_remove(idx);
                sorted.push(best_node.clone());

                #[allow(clippy::single_match)]
                match &nodes[best_node.index] {
                    RenderingPrimitive::UninitializedTexture(size) => {
                        logical_to_physical_render_targets[usages[best_node.index].logical_render_target]
                            .get_or_insert_with(|| {
                                self.render_texture_cache.temporary_render_texture(
                                    self.device,
                                    *size,
                                    crate::config::TEXTURE_FORMAT,
                                    usages[best_node.index].requires_mipmaps,
                                    usages[best_node.index].usages,
                                )
                            });
                    }
                    _ => {}
                }
                for parent in nodes[best_node.index].reads() {
                    let idx = topgraph[parent.index]
                        .remaining_reads_from_self
                        .iter()
                        .position(|x| *x == best_node)
                        .unwrap();
                    topgraph[parent.index].remaining_reads_from_self.swap_remove(idx);

                    if topgraph[parent.index].remaining_reads_from_self.is_empty() {
                        if let Some(other) = topgraph[parent.index].remaining_readwrite_from_self.clone() {
                            topgraph[other.index].remaining_dependencies -= 1;
                            if topgraph[other.index].remaining_dependencies == 0 {
                                available_nodes.push(other);
                            }
                        }
                    }

                    let rt = usages[parent.index].logical_render_target;
                    reduce_rt_refcount(
                        rt,
                        &mut logical_render_target_refcount,
                        &mut self.render_texture_cache,
                        &logical_to_physical_render_targets,
                    );
                }

                for write in nodes[best_node.index].writes() {
                    let rt = usages[write.index].logical_render_target;
                    reduce_rt_refcount(
                        rt,
                        &mut logical_render_target_refcount,
                        &mut self.render_texture_cache,
                        &logical_to_physical_render_targets,
                    );
                }

                for other in topgraph[best_node.index].remaining_reads_from_self.clone() {
                    // In order to resolve our current render target we need to allocate the render target for `other`.
                    // So prioritize that.
                    logical_render_target_pressure[usages[other.index].logical_render_target] += 1;
                    topgraph[other.index].remaining_dependencies -= 1;
                    if topgraph[other.index].remaining_dependencies == 0 {
                        available_nodes.push(other);
                    }
                }
                if topgraph[best_node.index].remaining_reads_from_self.is_empty() {
                    if let Some(other) = topgraph[best_node.index].remaining_readwrite_from_self.clone() {
                        topgraph[other.index].remaining_dependencies -= 1;
                        if topgraph[other.index].remaining_dependencies == 0 {
                            available_nodes.push(other);
                        }
                    }
                }
            } else {
                panic!("Deadlock in topological sort");
            }
        }

        let logical_to_physical_render_targets = logical_to_physical_render_targets
            .into_iter()
            .map(Option::unwrap)
            .collect::<Vec<_>>();
        (sorted, logical_to_physical_render_targets)

        // 1. Define graph
        // 2. Allocation buckets
    }

    fn build_top(
        &mut self,
        nodes: &[RenderingPrimitive],
        sorting: &[GraphNode],
        usages: &[Usage],
        passes: &mut Vec<CompiledPass>,
        logical_to_physical_render_targets: &[RenderTexture],
    ) {
        for node in sorting {
            let target_texture = &logical_to_physical_render_targets[usages[node.index].logical_render_target];
            match &nodes[node.index] {
                RenderingPrimitive::UninitializedTexture(_) => {
                    // Noop
                    // passes.push(CompiledPass {
                    //     clear: None,
                    //     ops: vec![],
                    // });
                    // passes.len() - 1
                }
                RenderingPrimitive::Clear(_, color) => {
                    passes.push(CompiledPass::RenderPass {
                        target: logical_to_physical_render_targets[usages[node.index].logical_render_target].clone(),
                        clear: Some(LoadOp::Clear(*color)),
                        ops: vec![],
                    });
                }
                RenderingPrimitive::Blit(Blit {
                    target,
                    source,
                    source_rect,
                    target_rect,
                    blend,
                }) => {
                    let source_texture_size = usages[source.index].size;
                    let target_texture_size = usages[target.index].size;
                    let target_texture_rect = CanvasRect::from_size(target_texture_size.to_f32().cast_unit());
                    let pipeline = self
                        .render_pipeline_cache
                        .get(
                            self.device,
                            RenderPipelineKey {
                                base: self.blitter.render_pipeline_base.clone().into(),
                                sample_count: target_texture.sample_count(),
                                depth_format: None,
                                target_format: target_texture.format(),
                                blend_state: *blend,
                            },
                        )
                        .to_owned();
                    let source_texture =
                        &logical_to_physical_render_targets[usages[source.index].logical_render_target];

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

                    Self::push_render_op_top(passes, target_texture, op);
                }
                RenderingPrimitive::Quad {
                    target,
                    rect,
                    pipeline,
                    material,
                } => {
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
                                    blend_state: material.blend,
                                },
                            )
                            .to_owned(),
                        bind_group: material.bind_group().to_owned(),
                    };

                    Self::push_render_op_top(passes, target_texture, op);
                }
                RenderingPrimitive::Mesh {
                    target,
                    vbo,
                    ibo,
                    pipeline,
                    material,
                } => {
                    let op = CompiledRenderingPrimitive::Render {
                        vbo: vbo.to_owned(),
                        ibo: ibo.to_owned(),
                        pipeline: self
                            .render_pipeline_cache
                            .get(
                                self.device,
                                RenderPipelineKey {
                                    base: pipeline.to_owned().into(),
                                    sample_count: target_texture.sample_count(),
                                    depth_format: None,
                                    target_format: target_texture.format(),
                                    blend_state: material.blend,
                                },
                            )
                            .to_owned(),
                        bind_group: material.bind_group().to_owned(),
                    };
                    Self::push_render_op_top(passes, target_texture, op);
                }
                RenderingPrimitive::GenerateMipmaps { target } => {
                    let bind_groups = (1..target_texture.mip_level_count())
                        .map(|mip_level| {
                            puffin::profile_scope!("create bind group");
                            let mat = self.material_cache.override_material(
                                self.device,
                                &self.mipmapper.material,
                                &[
                                    BindGroupEntryArc {
                                        binding: 0,
                                        resource: BindingResourceArc::Mipmap(Some((
                                            target_texture.clone(),
                                            mip_level - 1,
                                        ))),
                                    },
                                    BindGroupEntryArc {
                                        binding: 1,
                                        resource: BindingResourceArc::Mipmap(Some((target_texture.clone(), mip_level))),
                                    },
                                ],
                            );

                            mat.bind_group().to_owned()
                        })
                        .collect::<Vec<_>>();

                    Self::push_compute_op_top(
                        passes,
                        CompiledComputePrimitive::GenerateMipmaps {
                            target: target_texture.to_owned(),
                            pipeline: self.mipmapper.pipeline.clone(),
                            bind_groups,
                        },
                    )
                }
            }
        }
    }

    pub fn render(&mut self, passes: &[CompiledPass]) {
        puffin::profile_function!();
        for pass in passes {
            match pass {
                CompiledPass::RenderPass { target, clear, ops } => {
                    let color_attachment = wgpu::RenderPassColorAttachment {
                        view: target.get_mip_level_view(0).unwrap().view,
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
                                let index_count = ibo.size() as u32 / std::mem::size_of::<u32>() as u32;
                                wgpu_profiler!("op:render", self.gpu_profiler, &mut render_pass, &self.device, {
                                    render_pass.set_pipeline(&pipeline.pipeline);
                                    render_pass.set_bind_group(0, bind_group, &[]);
                                    render_pass.set_index_buffer(ibo.as_slice(), wgpu::IndexFormat::Uint32);
                                    render_pass.set_vertex_buffer(0, vbo.as_slice());
                                    render_pass.draw_indexed(0..index_count, 0, 0..1);
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
