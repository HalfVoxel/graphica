use core::panic;
use std::{
    collections::VecDeque,
    rc::Rc,
    sync::{Arc, Mutex},
};

use euclid::{default::Size2D, Point2D};
use lyon::math::{point, size};
use wgpu::{
    util::StagingBelt, BindGroup, BindGroupLayout, BlendState, BufferUsages, Color, CommandEncoder, ComputePipeline,
    Device, Extent3d, LoadOp, Origin3d, TextureFormat, TextureUsages,
};
use wgpu_profiler::{wgpu_profiler, GpuProfiler};

use crate::{
    blitter::{BlitGpuVertex, Blitter},
    cache::ephermal_buffer_cache::{BufferRange, EphermalBufferCache},
    cache::material_cache::{BindGroupEntryArc, BindingResourceArc, MaterialCache},
    cache::render_texture_cache::RenderTextureCache,
    cache::{
        material_cache::{DynamicMaterial, Material},
        render_pipeline_cache::{CachedRenderPipeline, RenderPipelineBase, RenderPipelineCache, RenderPipelineKey},
    },
    geometry_utilities::types::{CanvasRect, UVRect},
    mipmap::Mipmapper,
    texture::RenderTexture,
};

#[derive(Debug)]
struct Blit {
    source: GraphNode,
    target: GraphNode,
    source_rect: CanvasRect,
    target_rect: CanvasRect,
    blend: BlendState,
}

#[derive(Clone, Debug)]
pub enum RenderGraphMaterial {
    Material(Arc<Material>),
    DynamicMaterial(DynamicMaterial),
}

impl From<Arc<Material>> for RenderGraphMaterial {
    fn from(material: Arc<Material>) -> Self {
        Self::Material(material)
    }
}

impl From<DynamicMaterial> for RenderGraphMaterial {
    fn from(material: DynamicMaterial) -> Self {
        Self::DynamicMaterial(material)
    }
}

#[derive(Default)]
pub struct RenderGraph {
    nodes: Vec<RenderingPrimitive>,
}

#[derive(Debug)]
enum RenderingPrimitive {
    UninitializedTexture(Size2D<u32>, Option<TextureFormat>),
    UninitializedBuffer(usize),
    Clear(GraphNode, Color),
    Blit(Blit),
    Mesh {
        target: GraphNode,
        vbo: BufferRange,
        ibo: BufferRange,
        pipeline: Arc<RenderPipelineBase>,
        material: RenderGraphMaterial,
    },
    Quad {
        target: GraphNode,
        rect: CanvasRect,
        pipeline: Arc<RenderPipelineBase>,
        material: RenderGraphMaterial,
    },
    GenerateMipmaps {
        target: GraphNode,
    },
    Compute {
        target: GraphNode,
        pipeline: Arc<ComputePipeline>,
        material: RenderGraphMaterial,
        dispatch_size: (u32, u32, u32),
    },
    CustomCompute {
        writes: Vec<GraphNode>,
        reads: Vec<GraphNode>,
        f: CustomCompute,
    },
    CopyTextureToBuffer {
        source: GraphNode,
        target: GraphNode,
        extent: Extent3d,
    },
    OutputRenderTarget {
        target: GraphNode,
        render_texture: RenderTexture,
    },
    OutputBuffer {
        target: GraphNode,
        buffer: BufferRange,
    },
    Texture {
        texture: RenderTexture,
    },
    SnapshotTexture {
        source: GraphNode,
        texture: Arc<Mutex<Option<RenderTexture>>>,
    },
}

pub trait CustomComputePassPrimitive {
    fn compile<'a>(&'a self, context: &mut CompilationContext<'a>) -> Box<dyn CustomComputePass>;
}

pub trait CustomComputePass {
    fn execute<'a>(&'a self, device: &Device, gpu_profiler: &mut GpuProfiler, cpass: &mut wgpu::ComputePass<'a>);
}

#[derive(Clone)]
pub struct CustomCompute(Arc<dyn CustomComputePassPrimitive + 'static>);

impl std::fmt::Debug for CustomCompute {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CustomCompute")
    }
}

pub struct CompiledCustomCompute(Box<dyn CustomComputePass + 'static>);

impl std::fmt::Debug for CompiledCustomCompute {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CompiledCustomCompute")
    }
}

impl RenderingPrimitive {
    fn reads(&self) -> impl Iterator<Item = GraphNode> {
        match self {
            RenderingPrimitive::Blit(Blit { source, .. }) => vec![source.clone()].into_iter(),
            RenderingPrimitive::Quad { material, .. } | RenderingPrimitive::Mesh { material, .. } => match material {
                RenderGraphMaterial::Material(_) => vec![].into_iter(),
                RenderGraphMaterial::DynamicMaterial(m) => m
                    .overrides
                    .iter()
                    .filter_map(|e| match &e.resource {
                        BindingResourceArc::GraphNode(node) => Some(node.to_owned()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .into_iter(),
            },
            RenderingPrimitive::CustomCompute { reads, .. } => reads.clone().into_iter(),
            RenderingPrimitive::CopyTextureToBuffer { source, .. } => vec![source.clone()].into_iter(),
            RenderingPrimitive::SnapshotTexture { source, .. } => vec![source.clone()].into_iter(),
            _ => vec![].into_iter(),
        }
    }

    /// The parent resource which this node will render into
    fn target(&self) -> Option<GraphNode> {
        match self {
            RenderingPrimitive::UninitializedTexture(_, _) => None,
            RenderingPrimitive::Clear(target, _) => Some(target.clone()),
            RenderingPrimitive::Blit(Blit { target, .. }) => Some(target.clone()),
            RenderingPrimitive::Mesh { target, .. } => Some(target.clone()),
            RenderingPrimitive::Quad { target, .. } => Some(target.clone()),
            RenderingPrimitive::GenerateMipmaps { target } => Some(target.clone()),
            RenderingPrimitive::Compute { target, .. } => Some(target.clone()),
            RenderingPrimitive::CustomCompute { writes, .. } => writes.first().cloned(),
            RenderingPrimitive::UninitializedBuffer(_) => None,
            RenderingPrimitive::CopyTextureToBuffer { target, .. } => Some(target.clone()),
            RenderingPrimitive::OutputRenderTarget { target, .. } => Some(target.clone()),
            RenderingPrimitive::OutputBuffer { target, .. } => Some(target.clone()),
            RenderingPrimitive::Texture { .. } => None,
            RenderingPrimitive::SnapshotTexture { .. } => None,
        }
    }

    fn writes(&self) -> impl Iterator<Item = GraphNode> {
        if let RenderingPrimitive::Compute {
            material: RenderGraphMaterial::DynamicMaterial(m),
            ..
        } = self
        {
            // Assume all compute resources are read/write
            // TODO: Not necessarily true
            return m
                .overrides
                .iter()
                .filter_map(|e| match &e.resource {
                    BindingResourceArc::GraphNode(node) => Some(node.to_owned()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .into_iter();
        }

        if let RenderingPrimitive::CustomCompute { writes, .. } = self {
            return writes.clone().into_iter();
        }

        self.target().into_iter().collect::<Vec<_>>().into_iter()
    }
}

#[derive(Clone, PartialEq, Debug, Hash)]
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
    Compute {
        bind_group: Rc<BindGroup>,
        pipeline: Arc<ComputePipeline>,
        dispatch_size: (u32, u32, u32),
    },
    CustomCompute(CompiledCustomCompute),
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

#[derive(Debug)]
pub enum CompiledEncoderPrimitive {
    CopyTextureToBuffer {
        source: RenderTexture,
        target: BufferRange,
        extent: Extent3d,
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
        let tex = self.push_primitive(RenderingPrimitive::UninitializedTexture(size, None));
        self.push_primitive(RenderingPrimitive::Clear(tex, color))
    }

    pub fn clear_with_format(&mut self, size: Size2D<u32>, color: Color, format: TextureFormat) -> GraphNode {
        let tex = self.push_primitive(RenderingPrimitive::UninitializedTexture(size, Some(format)));
        self.push_primitive(RenderingPrimitive::Clear(tex, color))
    }

    pub fn mesh(
        &mut self,
        target: GraphNode,
        vbo: BufferRange,
        ibo: BufferRange,
        pipeline: Arc<RenderPipelineBase>,
        material: impl Into<RenderGraphMaterial>,
    ) -> GraphNode {
        self.push_primitive(RenderingPrimitive::Mesh {
            target,
            vbo,
            ibo,
            pipeline,
            material: material.into(),
        })
    }

    pub fn quad(
        &mut self,
        target: GraphNode,
        rect: CanvasRect,
        pipeline: Arc<RenderPipelineBase>,
        material: impl Into<RenderGraphMaterial>,
    ) -> GraphNode {
        self.push_primitive(RenderingPrimitive::Quad {
            target,
            rect,
            pipeline,
            material: material.into(),
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

    pub fn compute(
        &mut self,
        target: GraphNode,
        pipeline: Arc<ComputePipeline>,
        material: impl Into<RenderGraphMaterial>,
        dispatch_size: (u32, u32, u32),
    ) -> GraphNode {
        self.push_primitive(RenderingPrimitive::Compute {
            target,
            pipeline,
            material: material.into(),
            dispatch_size,
        })
    }

    pub fn generate_mipmaps(&mut self, target: GraphNode) -> GraphNode {
        self.push_primitive(RenderingPrimitive::GenerateMipmaps { target })
    }

    pub fn custom_compute(
        &mut self,
        reads: Vec<GraphNode>,
        writes: Vec<GraphNode>,
        pass: impl CustomComputePassPrimitive + 'static,
    ) -> GraphNode {
        self.push_primitive(RenderingPrimitive::CustomCompute {
            reads,
            writes,
            f: CustomCompute(Arc::new(pass)),
        })
    }

    pub fn output_texture(&mut self, target: GraphNode, texture: RenderTexture) {
        self.push_primitive(RenderingPrimitive::OutputRenderTarget {
            target,
            render_texture: texture,
        });
    }

    pub fn output_buffer(&mut self, target: GraphNode, buffer: BufferRange) {
        self.push_primitive(RenderingPrimitive::OutputBuffer { target, buffer });
    }

    pub fn uninitialized_buffer(&mut self, size: usize) -> GraphNode {
        self.push_primitive(RenderingPrimitive::UninitializedBuffer(size))
    }

    pub fn copy_texture_to_buffer(&mut self, source: GraphNode, target: GraphNode, extent: Extent3d) -> GraphNode {
        self.push_primitive(RenderingPrimitive::CopyTextureToBuffer { source, target, extent })
    }

    pub fn texture(&mut self, texture: RenderTexture) -> GraphNode {
        // TODO: Anything using this node as a target needs to insert a blit to a scratch texture first
        self.push_primitive(RenderingPrimitive::Texture { texture })
    }

    pub fn snapshot_texture(&mut self, source: GraphNode, texture: Arc<Mutex<Option<RenderTexture>>>) -> GraphNode {
        self.push_primitive(RenderingPrimitive::SnapshotTexture {
            source,
            texture: texture.clone(),
        })
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
    pub mipmapper: &'a Mipmapper,
    pub gpu_profiler: &'a mut GpuProfiler,
    pub pass_info_bind_group_layout: &'a BindGroupLayout,
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
    EncoderPass(CompiledEncoderPrimitive),
}

#[derive(Debug)]
struct RenderTextureHandle(usize);

#[derive(Debug, Clone)]
struct Usage {
    usages: TextureUsages,
    requires_mipmaps: bool,
    logical_resource: usize,
}

pub struct CompilationContext<'a> {
    pub device: &'a Device,
    material_cache: &'a mut MaterialCache,
    usages: &'a [Usage],
    logical_to_physical_resources: &'a [PhysicalResource],
}

impl<'a> CompilationContext<'a> {
    pub fn resolve_material(&mut self, material: &'a RenderGraphMaterial) -> &Arc<Material> {
        RenderGraphCompiler::resolve_dynamic_material(
            self.material_cache,
            material,
            self.usages,
            self.logical_to_physical_resources,
        )
    }
}

#[derive(Debug, Clone)]
enum PhysicalResource {
    RenderTexture(RenderTexture),
    Buffer(BufferRange),
}

impl PhysicalResource {
    pub fn expect_render_texture(&self) -> &RenderTexture {
        match self {
            PhysicalResource::RenderTexture(rt) => rt,
            PhysicalResource::Buffer(_) => panic!("Expected a render texture, but found a buffer"),
        }
    }

    pub fn expect_buffer(&self) -> &BufferRange {
        match self {
            PhysicalResource::RenderTexture(_) => panic!("Expected a buffer, but found a render texture"),
            PhysicalResource::Buffer(b) => b,
        }
    }
}

#[repr(C)]
#[repr(align(256))]
#[derive(Debug, Clone)]
pub struct ShaderPassInfo {
    render_target_size: Size2D<f32>,
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
            wgpu::BufferUsages::VERTEX,
            vertices,
        )
    }

    pub fn compile(&mut self, render_graph: &RenderGraph) -> Vec<CompiledPass> {
        puffin::profile_scope!("compile render graph");
        self.gpu_profiler.begin_scope("compile", self.encoder, self.device);
        let mut usages = vec![
            Usage {
                requires_mipmaps: false,
                usages: TextureUsages::empty(),
                logical_resource: usize::MAX,
            };
            render_graph.nodes.len()
        ];

        let nodes = &render_graph.nodes;
        let mut logical_render_target_index = 0;
        // Propagate backwards
        for i in (0..render_graph.nodes.len()).rev() {
            let mut usage = usages[i].clone();
            match &nodes[i] {
                RenderingPrimitive::UninitializedBuffer(_) | RenderingPrimitive::UninitializedTexture(_, _) => {
                    usage.logical_resource = logical_render_target_index;
                    logical_render_target_index += 1;
                }
                RenderingPrimitive::Clear(_, _) => {
                    usage.usages |= TextureUsages::RENDER_ATTACHMENT;
                }
                RenderingPrimitive::Blit(_) => {
                    usage.usages |= TextureUsages::RENDER_ATTACHMENT;
                }
                RenderingPrimitive::Quad { .. } => {
                    usage.usages |= TextureUsages::RENDER_ATTACHMENT;
                }
                RenderingPrimitive::Mesh { .. } => {
                    usage.usages |= TextureUsages::RENDER_ATTACHMENT;
                }
                RenderingPrimitive::GenerateMipmaps { .. } => {
                    usage.usages |= TextureUsages::STORAGE_BINDING;
                    usage.requires_mipmaps = true;
                }
                RenderingPrimitive::Compute { .. } => {
                    usage.usages |= TextureUsages::STORAGE_BINDING;
                }
                RenderingPrimitive::CustomCompute { .. } => {
                    usage.usages |= TextureUsages::STORAGE_BINDING;
                }
                RenderingPrimitive::CopyTextureToBuffer { .. } => {
                    // usage.usages |= TextureUsages::COPY_DST;
                }
                RenderingPrimitive::OutputRenderTarget { .. } => {
                    // NOP
                }
                RenderingPrimitive::OutputBuffer { .. } => {
                    // NOP
                }
                RenderingPrimitive::SnapshotTexture { .. } => {
                    usage.logical_resource = logical_render_target_index;
                    logical_render_target_index += 1;
                    // Seems reasonable?
                    usage.requires_mipmaps = true;
                    usage.usages |= TextureUsages::all()
                }
                RenderingPrimitive::Texture { .. } => {
                    usage.logical_resource = logical_render_target_index;
                    logical_render_target_index += 1;
                }
            }

            if let Some(target) = nodes[i].target() {
                usages[target.index].requires_mipmaps |= usage.requires_mipmaps;
                usages[target.index].usages |= usage.usages;
            }

            match &nodes[i] {
                RenderingPrimitive::Compute { .. } | RenderingPrimitive::CustomCompute { .. } => {
                    for write in nodes[i].writes() {
                        // Assume compute writes are storage textures. TODO: Not necessarily true.
                        usages[write.index].usages |= TextureUsages::STORAGE_BINDING;
                    }
                }
                _ => {}
            }

            for source in nodes[i].reads() {
                match &nodes[i] {
                    RenderingPrimitive::Compute { .. } | RenderingPrimitive::CustomCompute { .. } => {
                        // Assume compute reads are storage textures. TODO: Not necessarily true.
                        usages[source.index].usages |= TextureUsages::STORAGE_BINDING;
                    }
                    RenderingPrimitive::CopyTextureToBuffer { .. } => {
                        // Read is done using COPY
                        usages[source.index].usages |= TextureUsages::COPY_SRC;
                    }
                    _ => {
                        // Assume reads are sampled with texture samplers. TODO: Not necessarily true.
                        usages[source.index].usages |= TextureUsages::TEXTURE_BINDING;
                    }
                }
            }

            usages[i] = usage;
        }

        let mut logical_to_physical_resources = vec![None; logical_render_target_index];

        // Propagate forwards
        for i in 0..render_graph.nodes.len() {
            let mut usage = usages[i].clone();

            if let Some(target) = nodes[i].target() {
                usage.requires_mipmaps = usages[target.index].requires_mipmaps;
                usage.usages = usages[target.index].usages;
                usage.logical_resource = usages[target.index].logical_resource;
            }

            if usage.logical_resource == usize::MAX {
                panic!("Logical resource has not been set for {:#?}", nodes[i]);
            }

            // Assign pre-filled resources
            match &nodes[i] {
                RenderingPrimitive::OutputRenderTarget { render_texture, .. } => {
                    logical_to_physical_resources[usage.logical_resource] =
                        Some(PhysicalResource::RenderTexture(render_texture.clone()))
                }
                RenderingPrimitive::OutputBuffer { buffer, .. } => {
                    logical_to_physical_resources[usage.logical_resource] =
                        Some(PhysicalResource::Buffer(buffer.clone()))
                }
                RenderingPrimitive::Texture { texture } => {
                    logical_to_physical_resources[usage.logical_resource] =
                        Some(PhysicalResource::RenderTexture(texture.clone()))
                }
                _ => {}
            }

            usages[i] = usage;
        }

        for node in &render_graph.nodes {
            if let RenderingPrimitive::OutputRenderTarget { target, render_texture } = node {
                if !render_texture.usage().contains(usages[target.index].usages) {
                    panic!(
                            "The render target {:?} doesn't have sufficient usage flags. The render target has: {:?}, but required {:?}",
                            render_texture,
                            render_texture.usage(),
                            usages[target.index].usages
                        );
                }
            }
        }

        // logical_to_physical_resources[usages[source.index].logical_resource] = Some(target_texture.clone());

        let (sorted, logical_to_physical_render_targets) = {
            puffin::profile_scope!("schedule");
            self.topological_sort(&render_graph.nodes, &usages, logical_to_physical_resources)
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

    fn pixel_to_uv_rect(pixel_rect: &CanvasRect, texture_size: &Extent3d) -> UVRect {
        pixel_rect
            .scale(1.0 / (texture_size.width as f32), 1.0 / (texture_size.height as f32))
            .cast_unit()
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

    fn push_device_op_top(passes: &mut Vec<CompiledPass>, op: CompiledEncoderPrimitive) {
        passes.push(CompiledPass::EncoderPass(op));
    }

    fn resolve_dynamic_material<'b>(
        material_cache: &'b mut MaterialCache,
        material: &'b RenderGraphMaterial,
        usages: &[Usage],
        logical_to_physical_render_targets: &[PhysicalResource],
    ) -> &'b Arc<Material> {
        match material {
            RenderGraphMaterial::Material(material) => material,
            RenderGraphMaterial::DynamicMaterial(material) => {
                let overrides = material
                    .overrides
                    .iter()
                    .map(|e| BindGroupEntryArc {
                        binding: e.binding,
                        resource: match &e.resource {
                            BindingResourceArc::GraphNode(node) => BindingResourceArc::mipmap(Some((
                                logical_to_physical_render_targets[usages[node.index].logical_resource]
                                    .expect_render_texture()
                                    .clone(),
                                0,
                            ))),
                            x => x.clone(),
                        },
                    })
                    .collect::<Vec<_>>();
                material_cache.override_material(&material.material, &overrides)
            }
        }
    }

    fn topological_sort(
        &mut self,
        nodes: &[RenderingPrimitive],
        usages: &[Usage],
        // Mapping from logical render targets to physical render targets
        // May contain duplicates due to aliasing.
        mut logical_to_physical_resources: Vec<Option<PhysicalResource>>,
    ) -> (Vec<GraphNode>, Vec<PhysicalResource>) {
        #[derive(Default, Clone, Debug)]
        struct TopSortNode {
            remaining_dependencies: u32,
            remaining_reads_from_self: Vec<GraphNode>,
            remaining_readwrite_from_self: Option<GraphNode>,
            /// Tie-breaking bias. Higher values should be executed first.
            sort_order: u32,
            alive: bool,
        }

        let mut topgraph = vec![TopSortNode::default(); nodes.len()];
        let mut queued = vec![false; nodes.len()];
        let mut que = VecDeque::new();
        let mut sort_order_index = 0;
        // Refcounts for the logical render targets.
        // Predefined render textures start at a refcount of 1 so that they do not get deallocated
        let mut logical_render_target_refcount = logical_to_physical_resources
            .iter()
            .map(|x| x.is_some() as usize)
            .collect::<Vec<_>>();
        let mut original_refcounts = logical_render_target_refcount.clone();
        let mut logical_render_target_pressure = vec![0; logical_to_physical_resources.len()];

        for (i, node) in nodes.iter().enumerate() {
            if matches!(
                node,
                RenderingPrimitive::OutputBuffer { .. }
                    | RenderingPrimitive::OutputRenderTarget { .. }
                    | RenderingPrimitive::SnapshotTexture { .. }
            ) {
                que.push_front(GraphNode { index: i })
            }
        }

        while let Some(node) = que.pop_front() {
            topgraph[node.index].sort_order = sort_order_index;
            topgraph[node.index].alive = true;
            sort_order_index += 1;

            for target in nodes[node.index].reads() {
                logical_render_target_refcount[usages[target.index].logical_resource] += 1;

                topgraph[target.index].remaining_reads_from_self.push(node.clone());
                topgraph[node.index].remaining_dependencies += 1;
                if !queued[target.index] {
                    queued[target.index] = true;
                    que.push_back(target);
                }
            }
            for target in nodes[node.index].writes() {
                logical_render_target_refcount[usages[target.index].logical_resource] += 1;

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
            if node.remaining_dependencies == 0 && node.alive {
                available_nodes.push(GraphNode { index });
            }
        }

        fn reduce_rt_refcount(
            logical_render_target: usize,
            logical_render_target_refcount: &mut [usize],
            render_texture_cache: &mut RenderTextureCache,
            logical_to_physical_resources: &[Option<PhysicalResource>],
        ) {
            let refcount = &mut logical_render_target_refcount[logical_render_target];
            assert!(
                *refcount > 0,
                "Refcount for texture {:#?} went below zero",
                logical_to_physical_resources[logical_render_target]
            );
            *refcount -= 1;
            if *refcount == 0 {
                match &logical_to_physical_resources[logical_render_target] {
                    Some(PhysicalResource::RenderTexture(rt)) => {
                        // Deallocate
                        render_texture_cache.push(rt.to_owned());
                    }
                    Some(PhysicalResource::Buffer(_)) => {
                        // TODO
                    }
                    None => panic!("refcount reduced, but the resource has not been assigned"),
                }
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
                        RenderingPrimitive::UninitializedTexture(size, format) => self.render_texture_cache.has(
                            *size,
                            format.unwrap_or(crate::config::TEXTURE_FORMAT),
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
                            logical_render_target_pressure[usages[node.index].logical_resource],
                            topgraph[node.index].sort_order,
                        )
                    })
                });

            if let Some((idx, best_node)) = best_node {
                available_nodes.swap_remove(idx);
                sorted.push(best_node.clone());

                #[allow(clippy::single_match)]
                match &nodes[best_node.index] {
                    RenderingPrimitive::UninitializedTexture(size, format) => {
                        let resource = logical_to_physical_resources[usages[best_node.index].logical_resource]
                            .get_or_insert_with(|| {
                                PhysicalResource::RenderTexture(self.render_texture_cache.temporary_render_texture(
                                    self.device,
                                    *size,
                                    format.unwrap_or(crate::config::TEXTURE_FORMAT),
                                    usages[best_node.index].requires_mipmaps,
                                    usages[best_node.index].usages,
                                ))
                            });

                        // If the resource was pre-filled, we still want to ensure the type and size is correct
                        match resource {
                            PhysicalResource::RenderTexture(rt) => {
                                assert!(rt.size().width >= size.width);
                                assert!(rt.size().height >= size.height);
                            }
                            PhysicalResource::Buffer(_) => panic!("Expected a render texture"),
                        }
                    }
                    RenderingPrimitive::UninitializedBuffer(size) => {
                        let resource = logical_to_physical_resources[usages[best_node.index].logical_resource]
                            .get_or_insert_with(|| todo!());
                        // If the resource was pre-filled, we still want to ensure the type and size is correct
                        match resource {
                            PhysicalResource::RenderTexture(_) => {
                                panic!("Expected a buffer")
                            }
                            PhysicalResource::Buffer(b) => assert_eq!(b.size(), *size as u64),
                        }
                    }
                    RenderingPrimitive::SnapshotTexture { source, texture } => {
                        let parent = &logical_to_physical_resources[usages[source.index].logical_resource];
                        if let Some(PhysicalResource::RenderTexture(source_tex)) = parent {
                            let usage = &usages[best_node.index];
                            let mut guard = texture.lock().unwrap();
                            if let Some(tex) = &mut *guard {
                                if tex.usage().contains(usage.usages)
                                    && tex.size() == source_tex.size()
                                    && tex.format() == source_tex.format()
                                    && tex.mip_level_count() == source_tex.mip_level_count()
                                {
                                    // Still valid
                                } else {
                                    *guard = None;
                                }
                            }
                            let texture = (*guard).get_or_insert_with(|| {
                                self.render_texture_cache.temporary_render_texture(
                                    self.device,
                                    Size2D::new(source_tex.size().width, source_tex.size().height),
                                    source_tex.format(),
                                    true,
                                    usage.usages,
                                )
                            });
                            logical_to_physical_resources[usage.logical_resource] =
                                Some(PhysicalResource::RenderTexture(texture.clone()));
                            // Prevent it from being garbage collected
                            logical_render_target_refcount[usage.logical_resource] = 1;
                            original_refcounts[usage.logical_resource] = 1;
                        } else {
                            panic!(
                                "Source of persistent texture hasn't been set or is not a render texture: {:#?}",
                                parent
                            );
                        }
                    }
                    _ => {}
                }
                for parent in nodes[best_node.index].reads() {
                    let idx = topgraph[parent.index]
                        .remaining_reads_from_self
                        .iter()
                        .position(|x| *x == best_node)
                        .unwrap_or_else(|| {
                            panic!(
                                "Expected to find node {:#?} in the list of remaining reads from {:#?}",
                                nodes[best_node.index], nodes[parent.index]
                            )
                        });
                    topgraph[parent.index].remaining_reads_from_self.swap_remove(idx);

                    if topgraph[parent.index].remaining_reads_from_self.is_empty() {
                        if let Some(other) = topgraph[parent.index].remaining_readwrite_from_self.clone() {
                            topgraph[other.index].remaining_dependencies -= 1;
                            if topgraph[other.index].remaining_dependencies == 0 {
                                available_nodes.push(other);
                            }
                        }
                    }

                    // We are done reading these resources, so we can reduce their refcount
                    let rt = usages[parent.index].logical_resource;
                    reduce_rt_refcount(
                        rt,
                        &mut logical_render_target_refcount,
                        self.render_texture_cache,
                        &logical_to_physical_resources,
                    );
                }

                // We are done writing these targets, so we can reduce their refcount
                for write in nodes[best_node.index].writes() {
                    let rt = usages[write.index].logical_resource;
                    reduce_rt_refcount(
                        rt,
                        &mut logical_render_target_refcount,
                        self.render_texture_cache,
                        &logical_to_physical_resources,
                    );
                }

                for other in topgraph[best_node.index].remaining_reads_from_self.clone() {
                    // In order to resolve our current render target we need to allocate the render target for `other`.
                    // So prioritize that.
                    logical_render_target_pressure[usages[other.index].logical_resource] += 1;
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

        if sorted.len() < nodes.len() {
            dbg!(topgraph
                .iter()
                .enumerate()
                .zip(nodes.iter())
                .filter(|((i, _), _)| !sorted.contains(&GraphNode { index: *i }))
                .collect::<Vec<_>>());
            panic!(
                "Topological sort couldn't sort all nodes. Sorted {} out of {}",
                sorted.len(),
                nodes.len()
            );
        }

        let logical_to_physical_resources = logical_to_physical_resources
            .into_iter()
            .map(Option::unwrap)
            .collect::<Vec<_>>();

        if original_refcounts != logical_render_target_refcount {
            panic!(
                "Leaking render targets.\nExpected refcounts: {:?}, but found: {:?}.\nRender target allocation: {:?}",
                original_refcounts, logical_render_target_refcount, logical_to_physical_resources
            );
        }

        (sorted, logical_to_physical_resources)

        // 1. Define graph
        // 2. Allocation buckets
    }

    fn build_top(
        &mut self,
        nodes: &[RenderingPrimitive],
        sorting: &[GraphNode],
        usages: &[Usage],
        passes: &mut Vec<CompiledPass>,
        logical_to_physical_resources: &[PhysicalResource],
    ) {
        for node in sorting {
            let target_resource = &logical_to_physical_resources[usages[node.index].logical_resource];
            match &nodes[node.index] {
                RenderingPrimitive::UninitializedTexture(_, _) => {
                    // Noop
                    // passes.push(CompiledPass {
                    //     clear: None,
                    //     ops: vec![],
                    // });
                    // passes.len() - 1
                }
                RenderingPrimitive::UninitializedBuffer(_) => {
                    // Noop
                }
                RenderingPrimitive::Clear(_, color) => {
                    passes.push(CompiledPass::RenderPass {
                        target: logical_to_physical_resources[usages[node.index].logical_resource]
                            .expect_render_texture()
                            .clone(),
                        clear: Some(LoadOp::Clear(*color)),
                        ops: vec![],
                    });
                }
                RenderingPrimitive::Blit(Blit {
                    target: _,
                    source,
                    source_rect,
                    target_rect,
                    blend,
                }) => {
                    let target_texture = target_resource.expect_render_texture();
                    let source_texture =
                        logical_to_physical_resources[usages[source.index].logical_resource].expect_render_texture();
                    let source_texture_size = source_texture.size();
                    let target_texture_size = target_texture.size();
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

                    let mat = self.material_cache.override_material(
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
                        bind_group: mat.bind_group(self.device).to_owned(),
                        // bind_group: self.blit_bind_group(source_texture.default_view().view),
                        vbo: self.blit_vertices(
                            &Self::pixel_to_uv_rect(source_rect, &source_texture_size),
                            &Self::pixel_to_uv_rect(target_rect, &target_texture_size),
                        ),
                    };

                    Self::push_render_op_top(passes, target_texture, op);
                }
                RenderingPrimitive::Quad {
                    target: _,
                    rect,
                    pipeline,
                    material,
                } => {
                    let target_texture = target_resource.expect_render_texture();
                    let target_size = target_texture.size();

                    let uv_rect = Self::pixel_to_uv_rect(rect, &target_size);
                    let vbo = self.ephermal_buffer_cache.get(
                        self.device,
                        self.encoder,
                        self.staging_belt,
                        BufferUsages::VERTEX,
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
                        BufferUsages::INDEX,
                        &[0, 1, 2, 3, 2, 0],
                    );

                    let material = Self::resolve_dynamic_material(
                        self.material_cache,
                        material,
                        usages,
                        logical_to_physical_resources,
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
                        bind_group: material.bind_group(self.device).to_owned(),
                    };

                    Self::push_render_op_top(passes, target_texture, op);
                }
                RenderingPrimitive::Mesh {
                    target: _,
                    vbo,
                    ibo,
                    pipeline,
                    material,
                } => {
                    let target_texture = target_resource.expect_render_texture();
                    let material = Self::resolve_dynamic_material(
                        self.material_cache,
                        material,
                        usages,
                        logical_to_physical_resources,
                    );
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
                        bind_group: material.bind_group(self.device).to_owned(),
                    };
                    Self::push_render_op_top(passes, target_texture, op);
                }
                RenderingPrimitive::GenerateMipmaps { .. } => {
                    let target_texture = target_resource.expect_render_texture();
                    let bind_groups = (1..target_texture.mip_level_count())
                        .map(|mip_level| {
                            puffin::profile_scope!("create bind group");
                            let mat = self.material_cache.override_material(
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

                            mat.bind_group(self.device).to_owned()
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
                RenderingPrimitive::Compute {
                    pipeline,
                    material,
                    dispatch_size,
                    ..
                } => {
                    let material = Self::resolve_dynamic_material(
                        self.material_cache,
                        material,
                        usages,
                        logical_to_physical_resources,
                    );
                    Self::push_compute_op_top(
                        passes,
                        CompiledComputePrimitive::Compute {
                            bind_group: material.bind_group(self.device).clone(),
                            pipeline: pipeline.to_owned(),
                            dispatch_size: *dispatch_size,
                        },
                    )
                }
                RenderingPrimitive::CustomCompute { f, .. } => {
                    let mut context = CompilationContext {
                        device: self.device,
                        material_cache: self.material_cache,
                        usages,
                        logical_to_physical_resources,
                    };
                    Self::push_compute_op_top(
                        passes,
                        CompiledComputePrimitive::CustomCompute(CompiledCustomCompute(f.0.compile(&mut context))),
                    )
                }
                RenderingPrimitive::CopyTextureToBuffer { source, extent, .. } => {
                    let source = logical_to_physical_resources[usages[source.index].logical_resource]
                        .expect_render_texture()
                        .clone();
                    let target = target_resource.expect_buffer().clone();
                    Self::push_device_op_top(
                        passes,
                        CompiledEncoderPrimitive::CopyTextureToBuffer {
                            source,
                            target,
                            extent: *extent,
                        },
                    );
                }
                RenderingPrimitive::OutputRenderTarget { .. } => {
                    // Noop
                }
                RenderingPrimitive::OutputBuffer { .. } => {
                    // Noop
                }
                RenderingPrimitive::Texture { texture: _ } => {
                    // Noop
                }
                RenderingPrimitive::SnapshotTexture { source, texture: _ } => {
                    let target_texture = target_resource.expect_render_texture();
                    let source_texture =
                        logical_to_physical_resources[usages[source.index].logical_resource].expect_render_texture();
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

                    let mat = self.material_cache.override_material(
                        &self.blitter.material,
                        &[BindGroupEntryArc {
                            binding: 1,
                            resource: BindingResourceArc::render_texture(Some(source_texture.clone())),
                        }],
                    );

                    let op = CompiledRenderingPrimitive::Blit {
                        source: source_texture.clone(),
                        pipeline,
                        bind_group: mat.bind_group(self.device).to_owned(),
                        vbo: self.blit_vertices(
                            &UVRect::new(Point2D::new(0.0, 0.0), Size2D::new(1.0, 1.0).cast_unit()),
                            &UVRect::new(Point2D::new(0.0, 0.0), Size2D::new(1.0, 1.0).cast_unit()),
                        ),
                    };

                    Self::push_render_op_top(passes, target_texture, op);
                }
            }
        }
    }

    pub fn create_pass_infos(&mut self, passes: &[CompiledPass]) -> BindGroup {
        let pass_infos = passes
            .iter()
            .map(|pass| match pass {
                CompiledPass::RenderPass { target, .. } => ShaderPassInfo {
                    render_target_size: size(target.size().width as f32, target.size().height as f32),
                },
                _ => ShaderPassInfo {
                    render_target_size: size(0.0, 0.0),
                },
            })
            .collect::<Vec<_>>();

        let buffer = self.ephermal_buffer_cache.get(
            self.device,
            self.encoder,
            self.staging_belt,
            BufferUsages::UNIFORM,
            &pass_infos,
        );
        return self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pass infos"),
            layout: self.pass_info_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &buffer.buffer,
                    offset: buffer.range.start,
                    // Note: each bind group only binds 1 shader pass info, but the buffer contains many.
                    // Which shader pass info is bound is determined by the dynamic offset
                    size: wgpu::BufferSize::new(std::mem::size_of::<ShaderPassInfo>() as u64),
                }),
            }],
        });
    }

    pub fn render(&mut self, passes: &[CompiledPass]) {
        let pass_infos = self.create_pass_infos(passes);

        puffin::profile_function!();
        for (pass_index, pass) in passes.iter().enumerate() {
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
                    render_pass.set_bind_group(
                        1,
                        &pass_infos,
                        &[(pass_index * std::mem::size_of::<ShaderPassInfo>()) as u32],
                    );

                    for op in ops {
                        match op {
                            CompiledRenderingPrimitive::Blit {
                                source: _,
                                vbo,
                                pipeline,
                                bind_group,
                            } => {
                                puffin::profile_scope!("op:blit");
                                wgpu_profiler!("op:blit", self.gpu_profiler, &mut render_pass, self.device, {
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
                                wgpu_profiler!("op:render", self.gpu_profiler, &mut render_pass, self.device, {
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
                                wgpu_profiler!("op:generate_mipmaps", self.gpu_profiler, &mut cpass, self.device, {
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
                                        wgpu_profiler!("op:dispatch", self.gpu_profiler, &mut cpass, self.device, {
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
                            CompiledComputePrimitive::Compute {
                                bind_group,
                                pipeline,
                                dispatch_size,
                            } => {
                                wgpu_profiler!("op:compute", self.gpu_profiler, &mut cpass, self.device, {
                                    cpass.set_pipeline(pipeline);
                                    cpass.set_bind_group(0, bind_group, &[]);
                                    cpass.dispatch(dispatch_size.0, dispatch_size.1, dispatch_size.2);
                                });
                            }
                            CompiledComputePrimitive::CustomCompute(f) => {
                                f.0.execute(self.device, self.gpu_profiler, &mut cpass);
                            }
                        }
                    }

                    {
                        puffin::profile_scope!("drop cpass");
                        drop(cpass);
                    }
                }
                CompiledPass::EncoderPass(op) => {
                    match op {
                        CompiledEncoderPrimitive::CopyTextureToBuffer { source, target, extent } => {
                            let texture = match source {
                                RenderTexture::Texture(tex) => &***tex,
                                RenderTexture::SwapchainImage(_) => panic!("Cannot copy from swapchain images"),
                            };
                            self.encoder.copy_texture_to_buffer(
                                wgpu::ImageCopyTexture {
                                    texture: &texture.buffer,
                                    mip_level: 0,
                                    origin: Origin3d::ZERO,
                                    aspect: wgpu::TextureAspect::All,
                                },
                                wgpu::ImageCopyBuffer {
                                    buffer: &target.buffer,
                                    layout: wgpu::ImageDataLayout {
                                        offset: 0,
                                        // TODO: Only works on non-compressed textures
                                        bytes_per_row: Some(
                                            std::num::NonZeroU32::new(
                                                texture.descriptor.size.width
                                                    * texture.descriptor.format.describe().block_size as u32,
                                            )
                                            .unwrap(),
                                        ),
                                        rows_per_image: None,
                                    },
                                },
                                *extent,
                            );
                        }
                    }
                }
            }
        }
    }
}
