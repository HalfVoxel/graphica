use std::sync::{Arc, Mutex};

use cgmath::Matrix4;

use lyon::{
    lyon_tessellation::{BuffersBuilder, StrokeOptions, StrokeTessellator, VertexBuffers},
    path::Path,
};
use wgpu::{util::StagingBelt, BlendState, CommandEncoder, Device};
use winit::dpi::PhysicalSize;

use crate::{
    cache::{ephermal_buffer_cache::BufferRange, material_cache::Material, render_pipeline_cache::RenderPipelineBase},
    canvas::CanvasView,
    geometry_utilities::types::{CanvasLength, CanvasVector, ScreenLength},
    main::{Document, Globals, PosNormVertex, WithId},
    persistent_graph::{PersistentGraph, RenderNode},
    render_graph::GraphNode,
    wgpu_utils::{create_buffer_range, update_buffer_range_via_transfer},
};

#[repr(C, align(16))]
#[derive(Copy, Clone)]
pub struct Primitive {
    pub mvp_matrix: Matrix4<f32>,
    pub color: [f32; 4],
    pub width: f32,
}

pub struct VectorRenderer {
    pub target: Arc<Mutex<dyn RenderNode>>,
    vbo: BufferRange,
    ibo: BufferRange,
    scene_ubo: BufferRange,
    #[allow(dead_code)]
    primitive_ubo: BufferRange,
    vector_material: Arc<Material>,
    // index_buffer_length: usize,
    render_pipeline: Arc<RenderPipelineBase>,
}

impl VectorRenderer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        target: Arc<Mutex<dyn RenderNode>>,
        document: &Document,
        view: &CanvasView,
        device: &Device,
        encoder: &mut CommandEncoder,
        staging_belt: &mut StagingBelt,
        bind_group_layout: &Arc<wgpu::BindGroupLayout>,
        wireframe: bool,
        render_pipeline: &Arc<RenderPipelineBase>,
        wireframe_render_pipeline: &Arc<RenderPipelineBase>,
    ) -> Self {
        puffin::profile_function!();

        let mut builder = Path::builder();
        document.build(&mut builder);
        let p = builder.build();
        let mut canvas_tolerance: CanvasLength = ScreenLength::new(0.1) * view.screen_to_canvas_scale();
        let mut geometry: VertexBuffers<PosNormVertex, u32> = VertexBuffers::new();

        // It's important to clamp the tolerance to a not too small value
        // If the tesselator is fed a too small value it may get stuck in an infinite loop due to floating point precision errors
        canvas_tolerance = CanvasLength::new(canvas_tolerance.get().max(0.001));
        let canvas_line_width = ScreenLength::new(4.0) * view.screen_to_canvas_scale();
        StrokeTessellator::new()
            .tessellate_path(
                &p,
                &StrokeOptions::tolerance(canvas_tolerance.get()).with_line_width(canvas_line_width.get()),
                &mut BuffersBuilder::new(&mut geometry, WithId(0)),
            )
            .unwrap();

        let vbo = create_buffer_range(device, &geometry.vertices, wgpu::BufferUsages::VERTEX, "Document VBO");
        let ibo = create_buffer_range(device, &geometry.indices, wgpu::BufferUsages::INDEX, "Document IBO");

        let scene_ubo = create_buffer_range(
            device,
            &[Globals {
                resolution: [view.resolution.width as f32, view.resolution.height as f32],
                zoom: view.zoom,
                scroll_offset: view.scroll.to_array(),
            }],
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            "Document UBO",
        );

        let view_matrix = view.canvas_to_view_matrix();

        let primitive_ubo = create_buffer_range(
            device,
            &[Primitive {
                color: [1.0, 1.0, 1.0, 1.0],
                mvp_matrix: view_matrix * Matrix4::from_translation([0.0, 0.0, 0.1].into()),
                width: 0.0,
            }],
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            "Document Primitive UBO",
        );

        let vector_material = Arc::new(Material::from_consecutive_entries(
            "Vector",
            BlendState::ALPHA_BLENDING,
            bind_group_layout.to_owned(),
            vec![
                crate::cache::material_cache::BindingResourceArc::buffer(Some(scene_ubo.clone())),
                crate::cache::material_cache::BindingResourceArc::buffer(Some(primitive_ubo.clone())),
            ],
        ));

        let mut res = VectorRenderer {
            vbo,
            ibo,
            scene_ubo,
            primitive_ubo,
            vector_material,
            render_pipeline: if !wireframe {
                render_pipeline.to_owned()
            } else {
                wireframe_render_pipeline.to_owned()
            },
            target,
        };
        res.update(device, encoder, staging_belt);
        res
    }
}

impl RenderNode for VectorRenderer {
    fn update(&mut self, device: &Device, encoder: &mut CommandEncoder, staging_belt: &mut StagingBelt) {
        let view = CanvasView {
            zoom: 1.0,
            scroll: CanvasVector::new(0.0, 0.0),
            resolution: PhysicalSize::new(1024, 1024),
        };

        // TODO: Verify expected size?
        update_buffer_range_via_transfer(
            device,
            encoder,
            staging_belt,
            &[Globals {
                resolution: [view.resolution.width as f32, view.resolution.height as f32],
                zoom: view.zoom,
                scroll_offset: view.scroll.to_array(),
            }],
            &self.scene_ubo,
        );
    }

    fn render_node(&self, graph: &mut PersistentGraph) -> GraphNode {
        let target = graph.render(&self.target);
        graph.render_graph.mesh(
            target,
            self.vbo.clone(),
            self.ibo.clone(),
            self.render_pipeline.clone(),
            self.vector_material.clone(),
        )
    }
}
