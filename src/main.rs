use lyon::math::*;
use lyon::path::builder::*;
use lyon::path::Path;
use lyon::tessellation;
use lyon::tessellation::geometry_builder::*;
use lyon::tessellation::{FillOptions};
use lyon::tessellation::{StrokeOptions, StrokeTessellator};

use euclid;
use std::time::Instant;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, Event, KeyboardInput, MouseButton, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::Window;
use wgpu::{Device, Buffer, BindGroup, CommandEncoder, RenderPass, RenderPipeline};
use std::rc::Rc;

use crate::canvas::CanvasView;
use crate::fps_limiter::FPSLimiter;
use crate::geometry_utilities::types::*;
use crate::geometry_utilities::{ParamCurveDistanceEval};
use crate::input::{InputManager, KeyCombination};
use crate::path::*;
use crate::path_collection::{
    ControlPointReference, MutableReferenceResolver, PathCollection, ReferenceResolver, SelectionReference,
    VertexReference, PathReference
};
use crate::geometry_utilities;
use crate::gui;
use crate::toolbar::GUIRoot;
use cpuprofiler::PROFILER;
use crate::path_editor::*;
use crate::brush_editor::{BrushEditor, BrushData};
use kurbo::Point as KurboPoint;
use kurbo::CubicBez;
use async_std::task;
use palette::Srgba;
use palette::Pixel;
use arrayvec::ArrayVec;
use crate::wgpu_utils::*;
use crate::blitter::Blitter;
use crate::shader::load_shader;

#[repr(C)]
#[derive(Copy, Clone, Default)]
struct Globals {
    resolution: [f32; 2],
    scroll_offset: [f32; 2],
    zoom: f32,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct GpuVertex {
    position: [f32; 2],
    normal: [f32; 2],
    prim_id: i32,
}

type GPURGBA = [u8; 4];

#[repr(C)]
#[derive(Copy, Clone)]
struct BrushGpuVertex {
    position: CanvasPoint,
    uv: Point,
    color: GPURGBA,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct CloneBrushGpuVertex {
    position: CanvasPoint,
    uv_background_source: Point,
    uv_background_target: Point,
    uv_brush: Point,
    color: GPURGBA,
}


#[repr(C)]
#[derive(Copy, Clone)]
struct Primitive {
    color: [f32; 4],
    translate: [f32; 2],
    z_index: i32,
    width: f32,
}

const DEFAULT_WINDOW_WIDTH: u32 = 800;
const DEFAULT_WINDOW_HEIGHT: u32 = 800;

/// Creates a texture that uses MSAA and fits a given swap chain
fn create_multisampled_framebuffer(
    device: &wgpu::Device,
    sc_desc: &wgpu::SwapChainDescriptor,
    sample_count: u32,
) -> wgpu::TextureView {
    let multisampled_frame_descriptor = &wgpu::TextureDescriptor {
        size: wgpu::Extent3d {
            width: sc_desc.width,
            height: sc_desc.height,
            depth: 1,
        },
        mip_level_count: 1,
        array_layer_count: 1,
        sample_count: sample_count,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Bgra8Unorm,//sc_desc.format,
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        label: Some("Framebuffer"),
    };

    device
        .create_texture(multisampled_frame_descriptor)
        .create_default_view()
}

struct DocumentRenderer {
    vbo: Buffer,
    ibo: Buffer,
    scene_ubo: Buffer,
    primitive_ubo: Buffer,
    bind_group: BindGroup,
    index_buffer_length: usize,
    render_pipeline: Rc<RenderPipeline>,
    brush_renderer: BrushRenderer,
}

struct BrushRenderer {
    vbo: Buffer,
    ibo: Buffer,
    // ubo: Buffer,
    index_buffer_length: usize,
    bind_group: BindGroup,
    render_pipeline: Rc<RenderPipeline>,
}

fn sample_points_along_curve(path: &PathData, spacing: f32) -> Vec<(usize, CanvasPoint)> {
    let mut result = vec![];
    let mut builder = Path::builder();
    path.build(&mut builder);
    
    for sub_path in path.iter_sub_paths() {
        let mut offset = 0f32;

        for start in sub_path.iter_beziers() {
            let end = start.next().unwrap();
            let p0 = start.position();
            let p1 = start.control_after();
            let p2 = end.control_before();
            let p3 = end.position();
            let bezier = CubicBez::new(KurboPoint::new(p0.x as f64, p0.y as f64), KurboPoint::new(p1.x as f64, p1.y as f64), KurboPoint::new(p2.x as f64, p2.y as f64), KurboPoint::new(p3.x as f64, p3.y as f64));

            loop {
                match bezier.eval_at_distance((spacing + offset) as f64, 0.01) {
                    Ok(p) => {
                        result.push((start.index, point(p.x as f32, p.y as f32)));
                        offset += spacing;
                    }
                    Err(geometry_utilities::CurveTooShort { remaining }) => {
                        offset = remaining as f32 - spacing;
                        debug_assert!(remaining >= 0.0);
                        break;
                    }
                }
            }
        }
    }

    result
}

#[derive(Copy, Clone)]
struct BrushUniforms {
    _dummy: i32,
}



impl BrushRenderer {
    fn new(brush_data: &BrushData, _view: &CanvasView, device: &Device, encoder: &mut CommandEncoder, bind_group_layout: &wgpu::BindGroupLayout, scene_ubo: &Buffer, scene_ubo_size: u64, render_pipeline_brush: &Rc<RenderPipeline>, texture: &Texture) -> BrushRenderer {
        let size = brush_data.brush.size;
        let points = sample_points_along_curve(&brush_data.path, brush_data.brush.spacing * size);

        let vertices: Vec<BrushGpuVertex> = points.iter().flat_map(|&(vertex_index, pos)| {
            let color = brush_data.colors[vertex_index/3].into_format().into_raw();
            ArrayVec::from([
                BrushGpuVertex { position: pos + vector(-size, -size), uv: point(0.0, 0.0), color },
                BrushGpuVertex { position: pos + vector(size, -size), uv: point(1.0, 0.0), color },
                BrushGpuVertex { position: pos + vector(size, size), uv: point(1.0, 1.0), color },
                BrushGpuVertex { position: pos + vector(-size, size), uv: point(0.0, 1.0), color },
            ])
        }
        ).collect();

        let (vbo, _) = create_buffer_with_data(device, &vertices, wgpu::BufferUsage::VERTEX);
        
        let indices: Vec<u32> = (0..points.len() as u32).flat_map(|x| vec![
            4*x + 0, 4*x + 1, 4*x + 2,
            4*x + 3, 4*x + 2, 4*x + 0
        ]).collect();
        let (ibo, _) = create_buffer_with_data(device, &indices, wgpu::BufferUsage::INDEX);
        
        let (primitive_ubo, primitive_ubo_size) = create_buffer_via_transfer(device, encoder, &[BrushUniforms { _dummy: 0 }], wgpu::BufferUsage::UNIFORM, "Uniform buffer");

        // let primitive_ubo_transfer = device.create_buffer_with_data(as_u8_slice(&[BrushUniforms { dummy: 0 }]), wgpu::BufferUsage::COPY_SRC);
        // let primitive_ubo_size = std::mem::size_of::<BrushUniforms>() as u64;
        // let primitive_ubo = device.create_buffer(&wgpu::BufferDescriptor {
        //     size: primitive_ubo_size,
        //     usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        // });
        // encoder.copy_buffer_to_buffer(&primitive_ubo_transfer, 0, &primitive_ubo, 0, primitive_ubo_size);

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            lod_min_clamp: -100.0,
            lod_max_clamp: 100.0,
            compare: wgpu::CompareFunction::Always,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            label: Some("Brush bind group"),
            bindings: &[
                wgpu::Binding {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer {
                        buffer: &scene_ubo,
                        range: 0..scene_ubo_size,
                    },
                },
                wgpu::Binding {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer {
                        buffer: &primitive_ubo,
                        range: 0..primitive_ubo_size,
                    },
                },
                wgpu::Binding {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::Binding {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&texture.view),
                },
            ],
        });

        BrushRenderer {
            vbo,
            ibo,
            // ubo,
            index_buffer_length: indices.len(),
            bind_group,
            render_pipeline: render_pipeline_brush.clone()
        }
    }

    fn update(&mut self, view: &CanvasView, device: &Device, encoder: &mut CommandEncoder) {
        // let scene_ubo_transfer = device
        // .create_buffer_mapped(1, wgpu::BufferUsage::COPY_SRC)
        // .fill_from_slice(&[Globals {
        //     resolution: [view.resolution.width as f32, view.resolution.height as f32],
        //     zoom: view.zoom,
        //     scroll_offset: view.scroll.to_array(),
        // }]);

        // let scene_ubo_size = std::mem::size_of::<Globals>() as u64;
        // encoder.copy_buffer_to_buffer(&scene_ubo_transfer, 0, &self.scene_ubo, 0, scene_ubo_size);
    }

    fn render<'a>(&'a self, pass: &mut RenderPass<'a>) {
        pass.set_pipeline(&self.render_pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_index_buffer(&self.ibo, 0, 0);
        pass.set_vertex_buffer(0, &self.vbo, 0, 0);
        pass.draw_indexed(0..(self.index_buffer_length as u32), 0, 0..1);
    }
}

impl DocumentRenderer {
    fn new(document: &Document, view: &CanvasView, device: &Device, encoder: &mut CommandEncoder, bind_group_layout: &wgpu::BindGroupLayout, wireframe: bool, render_pipeline: &Rc<RenderPipeline>, wireframe_render_pipeline: &Rc<RenderPipeline>, render_pipeline_brush: &Rc<RenderPipeline>, bind_group_layout_brush: &wgpu::BindGroupLayout) -> DocumentRenderer {
        let mut builder = Path::builder();
        document.build(&mut builder);
        let p = builder.build();
        let mut canvas_tolerance: CanvasLength = ScreenLength::new(0.1) * view.screen_to_canvas_scale();
        let mut geometry: VertexBuffers<GpuVertex, u32> = VertexBuffers::new();

        // It's important to clamp the tolerance to a not too small value
        // If the tesselator is fed a too small value it may get stuck in an infinite loop due to floating point precision errors
        canvas_tolerance = CanvasLength::new(canvas_tolerance.get().max(0.001));
        let canvas_line_width = ScreenLength::new(4.0) * view.screen_to_canvas_scale();
        StrokeTessellator::new()
            .tessellate_path(
                &p,
                &StrokeOptions::tolerance(canvas_tolerance.get()).with_line_width(canvas_line_width.get()),
                &mut BuffersBuilder::new(&mut geometry, WithId(0 as i32)),
            )
            .unwrap();

        let (vbo, _) = create_buffer_via_transfer(device, encoder, &geometry.vertices, wgpu::BufferUsage::VERTEX, "Document VBO");

        let indices: Vec<u32> = if wireframe {
            // Transform the triangle primitives into line primitives: (0,1,2) => (0,1),(1,2),(2,0)
            geometry
                .indices
                .chunks_exact(3)
                .flat_map(|v| vec![v[0], v[1], v[1], v[2], v[2], v[0]])
                .collect()
        } else {
            geometry.indices
        };
        // last_index_count = indices.len();

        let (ibo, _) = create_buffer_via_transfer(device, encoder, &indices, wgpu::BufferUsage::INDEX, "Document IBO");
        
        let (scene_ubo, scene_ubo_size) = create_buffer_via_transfer(device, encoder, &[Globals {
            resolution: [view.resolution.width as f32, view.resolution.height as f32],
            zoom: view.zoom,
            scroll_offset: view.scroll.to_array(),
        }], wgpu::BufferUsage::UNIFORM, "Document UBO");

        let (primitive_ubo, primitive_ubo_size) = create_buffer_via_transfer(device, encoder, &[Primitive {
            color: [1.0, 1.0, 1.0, 1.0],
            translate: [0.0, 0.0],
            z_index: 100,
            width: 0.0,
        }], wgpu::BufferUsage::UNIFORM, "Document Primitive UBO");

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            label: Some("Document bind group"),
            bindings: &[
                wgpu::Binding {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer {
                        buffer: &scene_ubo,
                        range: 0..scene_ubo_size,
                    },
                },
                wgpu::Binding {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer {
                        buffer: &primitive_ubo,
                        range: 0..primitive_ubo_size,
                    },
                },
            ],
        });

        let brush_renderer = BrushRenderer::new(&document.brushes, view,device, encoder, bind_group_layout_brush, &scene_ubo, scene_ubo_size, render_pipeline_brush, &document.textures[0]);

        let mut res = DocumentRenderer {
            vbo,
            ibo,
            scene_ubo,
            primitive_ubo,
            bind_group,
            index_buffer_length: indices.len(),
            brush_renderer: brush_renderer,
            render_pipeline: if !wireframe { render_pipeline.clone() } else { wireframe_render_pipeline.clone() }
        };
        res.update(view, device, encoder);
        res
    }

    fn update(&mut self, view: &CanvasView, device: &Device, encoder: &mut CommandEncoder) {
        // TODO: Verify expected size?
        update_buffer_via_transfer(device, encoder, &[Globals {
            resolution: [view.resolution.width as f32, view.resolution.height as f32],
            zoom: view.zoom,
            scroll_offset: view.scroll.to_array(),
        }], &self.scene_ubo);
    }

    fn render<'a>(&'a self, pass: &mut RenderPass<'a>) {
        pass.set_pipeline(&self.render_pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_index_buffer(&self.ibo, 0, 0);
        pass.set_vertex_buffer(0, &self.vbo, 0, 0);
        pass.draw_indexed(0..(self.index_buffer_length as u32), 0, 0..1);
        self.brush_renderer.render(pass);
    }
}



pub fn main() {
    PROFILER.lock().unwrap().start("./my-prof.profile").expect("Couldn't start");

    println!("== wgpu example ==");
    println!("Controls:");
    println!("  Arrow keys: scrolling");
    println!("  PgUp/PgDown: zoom in/out");
    println!("  w: toggle wireframe mode");
    println!("  b: toggle drawing the background");
    println!("  a/z: increase/decrease the stroke width");

    let mut data = PathData::new();
    data.line_to(point(0.0, 0.0));
    data.line_to(point(10.0, 0.0));
    data.line_to(point(10.0, 10.0));
    data.close();

    // Number of samples for anti-aliasing
    // Set to 1 to disable
    let sample_count = 8;

    let t0 = Instant::now();

    let t1 = Instant::now();

    let t3 = Instant::now();

    let mut bg_geometry: VertexBuffers<Point, u16> = VertexBuffers::new();
    lyon::tessellation::basic_shapes::fill_rectangle(
        &Rect::new(point(-1.0 * 5.0, -1.0 * 5.0), size(2.0 * 5.0, 2.0 * 5.0)),
        &FillOptions::default(),
        &mut BuffersBuilder::new(&mut bg_geometry, Positions),
    )
    .unwrap();

    let t4 = Instant::now();

    println!("Loading svg: {:?}", (t1.duration_since(t0).as_secs_f32() * 1000.0));
    println!("Memory: {:?}", (t4.duration_since(t3).as_secs_f32() * 1000.0));

    let document = Document {
        paths: PathCollection { paths: vec![] },
        brushes: crate::brush_editor::BrushData::new(),
        textures: vec![],
    };
    let mut ui_document = Document {
        paths: PathCollection { paths: vec![] },
        brushes: crate::brush_editor::BrushData::new(),
        textures: vec![],
    };

    let mut gui = gui::Root::new();
    let gui_root = gui.add(GUIRoot::new());

    let mut editor = Editor {
        path_editor: PathEditor::new(&mut ui_document),
        brush_editor: BrushEditor::new(),
        document: document,
        ui_document: ui_document,
        gui: gui,
        gui_root: gui_root,
        scene: SceneParams {
            target_zoom: 1.0,
            view: CanvasView {
                zoom: 1.0,
                scroll: vector(0.0, 0.0),
                resolution: PhysicalSize::new(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT),
            },
            target_scroll: vector(0.0, 0.0),
            show_points: true,
            show_wireframe: false,
            stroke_width: 1.0,
            target_stroke_width: 1.0,
            draw_background: true,
            cursor_position: (0.0, 0.0),
            size_changed: true,
        },
        input: InputManager::new(),
    };

    let adapter = task::block_on(wgpu::Adapter::request(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None, // TODO
    },
        wgpu::BackendBit::PRIMARY
    )).unwrap();

    let (device, queue) = task::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        extensions: wgpu::Extensions {
            anisotropic_filtering: false,
        },
        limits: wgpu::Limits::default(),
    }));
    
    let vs_module = load_shader(&device, include_bytes!("./../shaders/geometry.vert.spv"));
    let fs_module = load_shader(&device, include_bytes!("./../shaders/geometry.frag.spv"));
    let bg_vs_module = load_shader(&device, include_bytes!("./../shaders/background.vert.spv"));
    let bg_fs_module = load_shader(&device, include_bytes!("./../shaders/background.frag.spv"));
    let brush_vs_module = load_shader(&device, include_bytes!("./../shaders/brush.vert.spv"));
    let brush_fs_module = load_shader(&device, include_bytes!("./../shaders/brush.frag.spv"));
    let clone_brush_vs_module = load_shader(&device, include_bytes!("./../shaders/clone_brush.vert.spv"));
    let clone_brush_fs_module = load_shader(&device, include_bytes!("./../shaders/clone_brush.frag.spv"));

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Geometry Bind Group Layout"),
        bindings: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStage::VERTEX,
                ty: wgpu::BindingType::UniformBuffer { dynamic: false },
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStage::VERTEX,
                ty: wgpu::BindingType::UniformBuffer { dynamic: false },
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        bind_group_layouts: &[&bind_group_layout],
    });

    let depth_stencil_state = Some(wgpu::DepthStencilStateDescriptor {
        format: wgpu::TextureFormat::Depth32Float,
        depth_write_enabled: true,
        depth_compare: wgpu::CompareFunction::Greater,
        stencil_front: wgpu::StencilStateFaceDescriptor::IGNORE,
        stencil_back: wgpu::StencilStateFaceDescriptor::IGNORE,
        stencil_read_mask: 0,
        stencil_write_mask: 0,
    });

    let mut render_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
        layout: &pipeline_layout,
        vertex_stage: wgpu::ProgrammableStageDescriptor {
            module: &vs_module,
            entry_point: "main",
        },
        fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
            module: &fs_module,
            entry_point: "main",
        }),
        rasterization_state: Some(wgpu::RasterizationStateDescriptor {
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: wgpu::CullMode::None,
            depth_bias: 0,
            depth_bias_slope_scale: 0.0,
            depth_bias_clamp: 0.0,
        }),
        primitive_topology: wgpu::PrimitiveTopology::TriangleList,
        color_states: &[wgpu::ColorStateDescriptor {
            format: wgpu::TextureFormat::Bgra8Unorm,
            color_blend: wgpu::BlendDescriptor {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha_blend: wgpu::BlendDescriptor {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            write_mask: wgpu::ColorWrite::ALL,
        }],
        depth_stencil_state: depth_stencil_state.clone(),
        vertex_state: wgpu::VertexStateDescriptor {
            index_format: wgpu::IndexFormat::Uint32,
            vertex_buffers: &[wgpu::VertexBufferDescriptor {
                stride: std::mem::size_of::<GpuVertex>() as u64,
                step_mode: wgpu::InputStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttributeDescriptor {
                        offset: 0,
                        format: wgpu::VertexFormat::Float2,
                        shader_location: 0,
                    },
                    wgpu::VertexAttributeDescriptor {
                        offset: 8,
                        format: wgpu::VertexFormat::Float2,
                        shader_location: 1,
                    },
                    wgpu::VertexAttributeDescriptor {
                        offset: 16,
                        format: wgpu::VertexFormat::Int,
                        shader_location: 2,
                    },
                ],
            }],
        },
        sample_count: sample_count,
        sample_mask: !0,
        alpha_to_coverage_enabled: false,
    };

    let render_pipeline = Rc::new(device.create_render_pipeline(&render_pipeline_descriptor));

    // TODO: this isn't what we want: we'd need the equivalent of VK_POLYGON_MODE_LINE,
    // but it doesn't seem to be exposed by wgpu?
    render_pipeline_descriptor.primitive_topology = wgpu::PrimitiveTopology::LineList;
    let wireframe_render_pipeline = Rc::new(device.create_render_pipeline(&render_pipeline_descriptor));

    let bg_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        layout: &pipeline_layout,
        vertex_stage: wgpu::ProgrammableStageDescriptor {
            module: &bg_vs_module,
            entry_point: "main",
        },
        fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
            module: &bg_fs_module,
            entry_point: "main",
        }),
        rasterization_state: Some(wgpu::RasterizationStateDescriptor {
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: wgpu::CullMode::None,
            depth_bias: 0,
            depth_bias_slope_scale: 0.0,
            depth_bias_clamp: 0.0,
        }),
        primitive_topology: wgpu::PrimitiveTopology::TriangleList,
        color_states: &[wgpu::ColorStateDescriptor {
            format: wgpu::TextureFormat::Bgra8Unorm,
            color_blend: wgpu::BlendDescriptor::REPLACE,
            alpha_blend: wgpu::BlendDescriptor::REPLACE,
            write_mask: wgpu::ColorWrite::ALL,
        }],
        depth_stencil_state: depth_stencil_state.clone(),
        vertex_state: wgpu::VertexStateDescriptor {
            index_format: wgpu::IndexFormat::Uint16,
            vertex_buffers: &[wgpu::VertexBufferDescriptor {
                stride: std::mem::size_of::<Point>() as u64,
                step_mode: wgpu::InputStepMode::Vertex,
                attributes: &[wgpu::VertexAttributeDescriptor {
                    offset: 0,
                    format: wgpu::VertexFormat::Float2,
                    shader_location: 0,
                }],
            }],
        },
        sample_count: sample_count,
        sample_mask: !0,
        alpha_to_coverage_enabled: false,
    });

    let bind_group_layout_brush = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Bind group layout brush"),
        bindings: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStage::VERTEX,
                ty: wgpu::BindingType::UniformBuffer { dynamic: false },
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStage::VERTEX,
                ty: wgpu::BindingType::UniformBuffer { dynamic: false },
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStage::FRAGMENT,
                ty: wgpu::BindingType::Sampler { comparison: false },
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStage::FRAGMENT,
                ty: wgpu::BindingType::SampledTexture { multisampled: true, dimension: wgpu::TextureViewDimension::D2, component_type: wgpu::TextureComponentType::Float },
            },
        ],
    });

    let pipeline_layout_brush = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        bind_group_layouts: &[&bind_group_layout_brush],
    });

    let render_pipeline_descriptor_brush = wgpu::RenderPipelineDescriptor {
        layout: &pipeline_layout_brush,
        vertex_stage: wgpu::ProgrammableStageDescriptor {
            module: &brush_vs_module,
            entry_point: "main",
        },
        fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
            module: &brush_fs_module,
            entry_point: "main",
        }),
        rasterization_state: Some(wgpu::RasterizationStateDescriptor {
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: wgpu::CullMode::None,
            depth_bias: 0,
            depth_bias_slope_scale: 0.0,
            depth_bias_clamp: 0.0,
        }),
        primitive_topology: wgpu::PrimitiveTopology::TriangleList,
        color_states: &[wgpu::ColorStateDescriptor {
            format: wgpu::TextureFormat::Bgra8Unorm,
            color_blend: wgpu::BlendDescriptor {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha_blend: wgpu::BlendDescriptor {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            write_mask: wgpu::ColorWrite::ALL,
        }],
        depth_stencil_state: Some(wgpu::DepthStencilStateDescriptor {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::Greater,
            stencil_front: wgpu::StencilStateFaceDescriptor::IGNORE,
            stencil_back: wgpu::StencilStateFaceDescriptor::IGNORE,
            stencil_read_mask: 0,
            stencil_write_mask: 0,
        }),
        vertex_state: wgpu::VertexStateDescriptor {
            index_format: wgpu::IndexFormat::Uint32,
            vertex_buffers: &[wgpu::VertexBufferDescriptor {
                stride: std::mem::size_of::<BrushGpuVertex>() as u64,
                step_mode: wgpu::InputStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttributeDescriptor {
                        offset: 0,
                        format: wgpu::VertexFormat::Float2,
                        shader_location: 0,
                    },
                    wgpu::VertexAttributeDescriptor {
                        offset: 8,
                        format: wgpu::VertexFormat::Float2,
                        shader_location: 1,
                    },
                    wgpu::VertexAttributeDescriptor {
                        offset: 16,
                        format: wgpu::VertexFormat::Uchar4Norm,
                        shader_location: 2,
                    },
                ],
            }],
        },
        sample_count: sample_count,
        sample_mask: !0,
        alpha_to_coverage_enabled: false,
    };

    let render_pipeline_brush = Rc::new(device.create_render_pipeline(&render_pipeline_descriptor_brush));


    let bind_group_layout_clone_brush = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Bind group layout clone_brush"),
        bindings: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStage::VERTEX,
                ty: wgpu::BindingType::UniformBuffer { dynamic: false },
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStage::VERTEX,
                ty: wgpu::BindingType::UniformBuffer { dynamic: false },
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStage::FRAGMENT,
                ty: wgpu::BindingType::Sampler { comparison: false },
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStage::FRAGMENT,
                ty: wgpu::BindingType::SampledTexture { multisampled: true, dimension: wgpu::TextureViewDimension::D2, component_type: wgpu::TextureComponentType::Float },
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStage::FRAGMENT,
                ty: wgpu::BindingType::SampledTexture { multisampled: true, dimension: wgpu::TextureViewDimension::D2, component_type: wgpu::TextureComponentType::Float },
            },
        ],
    });

    let pipeline_layout_clone_brush = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        bind_group_layouts: &[&bind_group_layout_clone_brush],
    });

    let render_pipeline_descriptor_clone_brush = wgpu::RenderPipelineDescriptor {
        layout: &pipeline_layout_clone_brush,
        vertex_stage: wgpu::ProgrammableStageDescriptor {
            module: &clone_brush_vs_module,
            entry_point: "main",
        },
        fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
            module: &clone_brush_fs_module,
            entry_point: "main",
        }),
        rasterization_state: Some(wgpu::RasterizationStateDescriptor {
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: wgpu::CullMode::None,
            depth_bias: 0,
            depth_bias_slope_scale: 0.0,
            depth_bias_clamp: 0.0,
        }),
        primitive_topology: wgpu::PrimitiveTopology::TriangleList,
        color_states: &[wgpu::ColorStateDescriptor {
            format: wgpu::TextureFormat::Bgra8Unorm,
            color_blend: wgpu::BlendDescriptor {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::Zero,
                operation: wgpu::BlendOperation::Add,
            },
            alpha_blend: wgpu::BlendDescriptor {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::Zero,
                operation: wgpu::BlendOperation::Add,
            },
            write_mask: wgpu::ColorWrite::ALL,
        }],
        depth_stencil_state: None,
        vertex_state: wgpu::VertexStateDescriptor {
            index_format: wgpu::IndexFormat::Uint32,
            vertex_buffers: &[wgpu::VertexBufferDescriptor {
                stride: std::mem::size_of::<CloneBrushGpuVertex>() as u64,
                step_mode: wgpu::InputStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttributeDescriptor {
                        offset: 0,
                        format: wgpu::VertexFormat::Float2,
                        shader_location: 0,
                    },
                    wgpu::VertexAttributeDescriptor {
                        offset: 8,
                        format: wgpu::VertexFormat::Float2,
                        shader_location: 1,
                    },
                    wgpu::VertexAttributeDescriptor {
                        offset: 16,
                        format: wgpu::VertexFormat::Float2,
                        shader_location: 2,
                    },
                    wgpu::VertexAttributeDescriptor {
                        offset: 24,
                        format: wgpu::VertexFormat::Float2,
                        shader_location: 3,
                    },
                    wgpu::VertexAttributeDescriptor {
                        offset: 32,
                        format: wgpu::VertexFormat::Uchar4Norm,
                        shader_location: 4,
                    },
                ],
            }],
        },
        sample_count: 1,
        sample_mask: !0,
        alpha_to_coverage_enabled: false,
    };

    let render_pipeline_clone_brush = Rc::new(device.create_render_pipeline(&render_pipeline_descriptor_clone_brush));


    let event_loop = EventLoop::new();
    let window = Window::new(&event_loop).unwrap();
    let size = window.inner_size();

    let mut swap_chain_desc = wgpu::SwapChainDescriptor {
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        format: wgpu::TextureFormat::Bgra8Unorm,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Fifo,
    };

    let mut multisampled_render_target = None;

    let window_surface = wgpu::Surface::create(&window);
    let mut swap_chain = device.create_swap_chain(&window_surface, &swap_chain_desc);

    let mut depth_texture_view = None;

    let mut frame_count: f32 = 0.0;
    let mut last_time = Instant::now();
    let mut fps_limiter = FPSLimiter::new();
    let mut last_hash1 = 0u64;
    let mut last_hash2 = 0u64;
    let mut document_renderer1 = None;
    let mut document_renderer2 = None;

    let mut init_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Init encoder") });

    let blitter = Blitter::new(&device, &mut init_encoder);

    let (bg_vbo, _) = create_buffer_via_transfer(&device, &mut init_encoder, &bg_geometry.vertices, wgpu::BufferUsage::VERTEX, "BG VBO");
    let (bg_ibo, _) = create_buffer_via_transfer(&device, &mut init_encoder, &bg_geometry.indices, wgpu::BufferUsage::INDEX, "BG IBO");

    let bg_ubo_data = &[
        Primitive {
            color: [1.0, 1.0, 1.0, 1.0],
            translate: [0.0, 0.0],
            z_index: 100,
            width: 0.0,
        }
    ];
    let (bg_ubo, bg_ubo_size) = create_buffer_via_transfer(&device, &mut init_encoder, bg_ubo_data, wgpu::BufferUsage::UNIFORM, "BG UBO");

    let (globals_ubo, globals_ubo_size) = create_buffer_via_transfer(&device, &mut init_encoder, &[Globals { ..Default::default() }], wgpu::BufferUsage::UNIFORM, "Globals UBO");

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        label: Some("Background Bind Group"),
        bindings: &[
            wgpu::Binding {
                binding: 0,
                resource: wgpu::BindingResource::Buffer {
                    buffer: &globals_ubo,
                    range: 0..globals_ubo_size,
                },
            },
            wgpu::Binding {
                binding: 1,
                resource: wgpu::BindingResource::Buffer {
                    buffer: &bg_ubo,
                    range: 0..bg_ubo_size,
                },
            },
        ],
    });

    let tex = std::sync::Arc::new(Texture::load_from_file(std::path::Path::new("brush.png"), &device, &mut init_encoder).unwrap());
    editor.document.textures.push(tex.clone());
    editor.ui_document.textures.push(tex.clone());

    queue.submit(&[init_encoder.finish()]);

    event_loop.run(move |event, _, control_flow| {
        let scene = &mut editor.scene;
        let new_time = Instant::now();
        let dt = (new_time.duration_since(last_time)).as_secs_f32();
        last_time = new_time;

        if update_inputs(event, control_flow, &mut editor.input, scene, dt) {
            // keep polling inputs.
            return;
        }

        editor.gui.update();
        editor.gui.input(&mut editor.ui_document, &mut editor.input);
        editor.gui.render(&mut editor.ui_document, &scene.view);
        // editor.toolbar.update_ui(&mut editor.ui_document, &mut editor.document, &scene.view, &mut editor.input);
        editor.path_editor.update(&mut editor.ui_document, &mut editor.document, &scene.view, &mut editor.input, &editor.gui_root.get(&editor.gui).tool);
        editor.brush_editor.update(&mut editor.ui_document, &mut editor.document, &scene.view, &mut editor.input, &editor.gui_root.get(&editor.gui).tool);

        if editor.input.on_combination(
            &KeyCombination::new()
                .and(VirtualKeyCode::LControl)
                .and(VirtualKeyCode::W),
        ) {
            // Quit
            *control_flow = ControlFlow::Exit;
            // PROFILER.lock().unwrap().stop().expect("Couldn't stop");
            return;
        }

        // println!("Path editor = {:?}", t0.elapsed());

        if scene.size_changed {
            scene.size_changed = false;
            let physical = scene.view.resolution;
            swap_chain_desc.width = physical.width;
            swap_chain_desc.height = physical.height;
            swap_chain = device.create_swap_chain(&window_surface, &swap_chain_desc);

            let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Framebuffer depth"),
                size: wgpu::Extent3d {
                    width: swap_chain_desc.width,
                    height: swap_chain_desc.height,
                    depth: 1,
                },
                mip_level_count: 1,
                sample_count: sample_count,
                array_layer_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
            });

            depth_texture_view = Some(depth_texture.create_default_view());

            multisampled_render_target = if sample_count > 1 {
                Some(create_multisampled_framebuffer(&device, &swap_chain_desc, sample_count))
            } else {
                None
            };
        }

        let frame = swap_chain.get_next_texture().unwrap();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Frame encoder") });

        update_buffer_via_transfer(&device, &mut encoder, &[Globals {
            resolution: [scene.view.resolution.width as f32, scene.view.resolution.height as f32],
            zoom: scene.view.zoom,
            scroll_offset: scene.view.scroll.to_array(),
        }], &globals_ubo);

        let hash = editor.document.hash() ^ scene.view.hash();
        if hash != last_hash1 || document_renderer1.is_none() || true {
            last_hash1 = hash;
            document_renderer1 = Some(DocumentRenderer::new(&editor.document, &scene.view, &device, &mut encoder, &bind_group_layout, scene.show_wireframe, &render_pipeline, &wireframe_render_pipeline, &render_pipeline_brush, &bind_group_layout_brush));
        }

        let ui_view = CanvasView {
            zoom: 1.0,
            scroll: vector(scene.view.resolution.width as f32, scene.view.resolution.height as f32) * 0.5,
            resolution: scene.view.resolution,
        };
        let hash = editor.ui_document.hash() ^ ui_view.hash();
        if hash != last_hash2 || document_renderer2.is_none() || true {
            last_hash2 = hash;
            document_renderer2 = Some(DocumentRenderer::new(&editor.ui_document, &ui_view, &device, &mut encoder, &bind_group_layout, scene.show_wireframe, &render_pipeline, &wireframe_render_pipeline, &render_pipeline_brush, &bind_group_layout_brush));
        }

        document_renderer1.as_mut().unwrap().update(&scene.view, &device, &mut encoder);
        document_renderer2.as_mut().unwrap().update(&ui_view, &device, &mut encoder);

        let texture_extent2 = wgpu::Extent3d {
            width: scene.view.resolution.width,
            height: scene.view.resolution.height,
            depth: 1,
        };

        // Render into this texture
        let temp_frame = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Temp frame texture"),
            size: texture_extent2,
            mip_level_count: 1,
            sample_count: 1,
            array_layer_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Bgra8Unorm,
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        });
        let temp_frame_view = temp_frame.create_default_view();

        {
            // A resolve target is only supported if the attachment actually uses anti-aliasing
            // So if sample_count == 1 then we must render directly to the swapchain's buffer
            let color_attachment = if let Some(msaa_target) = &multisampled_render_target {
                wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: msaa_target,
                    load_op: wgpu::LoadOp::Clear,
                    store_op: wgpu::StoreOp::Store,
                    clear_color: wgpu::Color::BLACK,
                    resolve_target: Some(&temp_frame_view),
                }
            } else {
                wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: &temp_frame_view,
                    load_op: wgpu::LoadOp::Clear,
                    store_op: wgpu::StoreOp::Store,
                    clear_color: wgpu::Color::WHITE,
                    resolve_target: None,
                }
            };

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[color_attachment],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachmentDescriptor {
                    attachment: depth_texture_view.as_ref().unwrap(),
                    depth_load_op: wgpu::LoadOp::Clear,
                    depth_store_op: wgpu::StoreOp::Store,
                    stencil_load_op: wgpu::LoadOp::Clear,
                    stencil_store_op: wgpu::StoreOp::Store,
                    clear_depth: 0.0,
                    clear_stencil: 0,
                }),
            });

            if scene.draw_background {
                pass.set_pipeline(&bg_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.set_index_buffer(&bg_ibo, 0, 0);
                pass.set_vertex_buffer(0, &bg_vbo, 0, 0);

                pass.draw_indexed(0..6, 0, 0..1);
            }

            document_renderer1.as_ref().unwrap().render(&mut pass);
            document_renderer2.as_ref().unwrap().render(&mut pass);
        }

        {
            let mut p = PathData::new();
            p.move_to(point(0.0, 0.0));
            p.line_to(point(500.0, 500.0));
            p.end();
            let points = sample_points_along_curve(&p, 1.0);

            let to_normalized_pos = |v: CanvasPoint| {
                editor.scene.view.screen_to_normalized(editor.scene.view.canvas_to_screen_point(v))
            };

            let size = 10.0;
            let color = Srgba::new(1.0, 1.0, 1.0, 1.0).into_format().into_raw();
            let vertices: Vec<CloneBrushGpuVertex> = points.iter().flat_map(|&(_vertex_index, pos)| {
                let clone_pos = pos + vector(-0.707, -0.707);
                ArrayVec::from([
                    CloneBrushGpuVertex { position: pos + vector(-size, -size), uv_background_source: to_normalized_pos(clone_pos + vector(-size, -size)), uv_background_target: to_normalized_pos(pos + vector(-size, -size)), uv_brush: point(0.0, 0.0), color },
                    CloneBrushGpuVertex { position: pos + vector(size, -size), uv_background_source: to_normalized_pos(clone_pos + vector(size, -size)), uv_background_target: to_normalized_pos(pos + vector(size, -size)), uv_brush: point(1.0, 0.0), color },
                    CloneBrushGpuVertex { position: pos + vector(size, size), uv_background_source: to_normalized_pos(clone_pos + vector(size, size)), uv_background_target: to_normalized_pos(pos + vector(size, size)), uv_brush: point(1.0, 1.0), color },
                    CloneBrushGpuVertex { position: pos + vector(-size, size), uv_background_source: to_normalized_pos(clone_pos + vector(-size, size)), uv_background_target: to_normalized_pos(pos + vector(-size, size)), uv_brush: point(0.0, 1.0), color },
                ])
            }
            ).collect();

            let (vbo, _) = create_buffer_with_data(&device, &vertices, wgpu::BufferUsage::VERTEX);
            
            let indices: Vec<u32> = (0..points.len() as u32).flat_map(|x| vec![
                4*x + 0, 4*x + 1, 4*x + 2,
                4*x + 3, 4*x + 2, 4*x + 0
            ]).collect();
            let (ibo, _) = create_buffer_with_data(&device, &indices, wgpu::BufferUsage::INDEX);

            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Linear,
                lod_min_clamp: -100.0,
                lod_max_clamp: 100.0,
                compare: wgpu::CompareFunction::Always,
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &bind_group_layout_clone_brush,
                label: Some("Clone brush Bind Group"),
                bindings: &[
                    wgpu::Binding {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer {
                            buffer: &globals_ubo,
                            range: 0..globals_ubo_size,
                        },
                    },
                    wgpu::Binding {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer {
                            buffer: &bg_ubo,
                            range: 0..bg_ubo_size,
                        },
                    },
                    wgpu::Binding {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                    wgpu::Binding {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&temp_frame_view),
                    },
                    wgpu::Binding {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&tex.view),
                    },
                ],
            });

            let size_in_pixels = (2.0*(CanvasLength::new(size) * editor.scene.view.canvas_to_screen_scale()).get()).ceil() as u32;

            let texture_extent = wgpu::Extent3d {
                width: 2*size_in_pixels + 2,
                height: 2*size_in_pixels + 2,
                depth: 1,
            };
            let temp_texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Temp texture"),
                size: texture_extent,
                mip_level_count: 1,
                sample_count: 1,
                array_layer_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Bgra8Unorm,
                usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::OUTPUT_ATTACHMENT,
            });
            
            // encoder.copy_texture_to_texture(
            //     wgpu::TextureCopyView { texture: &frame, mip_level: 0, array_layer: 0, origin: wgpu::Origin3d { x: 0, y: 0, z: 0}},
            //     wgpu::TextureCopyView { texture: &temp_frame, mip_level: 0, array_layer: 0, origin: wgpu::Origin3d { x: 0, y: 0, z: 0}},
            //     wgpu::Extent3d { width: editor.scene.view.resolution.width, height: editor.scene.view.resolution.height, depth: 1 }
            // );
    
            let texture_view = temp_texture.create_default_view();

            let temp_to_frame_blitter = blitter.with_textures(&device, &texture_view, &temp_frame_view);

            for i in 0..points.len() {
                {
                    let mut pass2 = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        color_attachments: &[
                            wgpu::RenderPassColorAttachmentDescriptor {
                                attachment: &texture_view,
                                load_op: wgpu::LoadOp::Clear,
                                store_op: wgpu::StoreOp::Store,
                                clear_color: wgpu::Color::RED,
                                resolve_target: None,
                            }
                        ],
                        depth_stencil_attachment: None,
                    });

                    pass2.set_pipeline(&render_pipeline_clone_brush);
                    pass2.set_bind_group(0, &bind_group, &[]);
                    pass2.set_index_buffer(&ibo, 0, 0);
                    pass2.set_vertex_buffer(0, &vbo, 0, 0);
                    pass2.draw_indexed((i * 6) as u32..((i+1) * 6) as u32, 0, 0..1);
                }

                {
                    let uv = vertices[(i*4)..(i+1)*4].iter().map(|x| x.uv_background_target).collect::<Vec<Point>>();
                    let r = Rect::from_points(uv);
                    temp_to_frame_blitter.blit(&device, &mut encoder, rect(0.0, 0.0, 1.0, 1.0), r);
                    // Blit pass
                    // let mut pass3 = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    //     color_attachments: &[
                    //         wgpu::RenderPassColorAttachmentDescriptor {
                    //             attachment: &frame.view,
                    //             load_op: wgpu::LoadOp::Load,
                    //             store_op: wgpu::StoreOp::Store,
                    //             clear_color: wgpu::Color::RED,
                    //             resolve_target: None,
                    //         }
                    //     ],
                    //     depth_stencil_attachment: None,
                    // });

                    // pass3.set_pipeline(&render_pipeline_clone_brush_blit);
                    // pass3.set_bind_group(0, &bind_group, &[]);
                    // pass3.set_index_buffer(&ibo, 0, 0);
                    // pass3.set_vertex_buffer(0, &vbo, 0, 0);
                    // pass3.draw_indexed((i * 6) as u32..((i+1) * 6) as u32, 0, 0..1);
                }
            }
        }

        blitter.blit(&device, &mut encoder, &temp_frame_view, &frame.view, rect(0.0, 0.0, 1.0, 1.0), rect(0.0, 1.0, 1.0, -1.0));

        queue.submit(&[encoder.finish()]);
        // dbg!(t3.elapsed());

        frame_count += 1.0;
        editor.input.tick_frame();
        // println!("Preparing GPU work = {:?}", t1.elapsed());
        fps_limiter.wait(std::time::Duration::from_secs_f32(1.0 / 60.0));
    });
}

/// This vertex constructor forwards the positions and normals provided by the
/// tessellators and add a shape id.
pub struct WithId(pub i32);

impl FillVertexConstructor<GpuVertex> for WithId {
    fn new_vertex(&mut self, position: Point, _attributes: tessellation::FillAttributes) -> GpuVertex {
        debug_assert!(!position.x.is_nan());
        debug_assert!(!position.y.is_nan());
        GpuVertex {
            position: position.to_array(),
            normal: [0.0, 0.0],
            prim_id: self.0,
        }
    }
}

impl StrokeVertexConstructor<GpuVertex> for WithId {
    fn new_vertex(&mut self, position: Point, attributes: tessellation::StrokeAttributes) -> GpuVertex {
        debug_assert!(!position.x.is_nan());
        debug_assert!(!position.y.is_nan());
        debug_assert!(!attributes.normal().x.is_nan());
        debug_assert!(!attributes.normal().y.is_nan());
        debug_assert!(!attributes.advancement().is_nan());
        GpuVertex {
            position: position.to_array(),
            normal: attributes.normal().to_array(),
            prim_id: self.0,
        }
    }
}

pub struct Editor {
    gui: gui::Root,
    gui_root: gui::TypedWidgetReference<crate::toolbar::GUIRoot>,
    path_editor: PathEditor,
    brush_editor: BrushEditor,
    scene: SceneParams,
    ui_document: Document,
    document: Document,
    input: InputManager,
}

pub struct Texture {
    pub buffer: wgpu::Texture,
    pub view: wgpu::TextureView,
}

impl Texture {
    fn load_from_file(path: &std::path::Path, device: &Device, encoder: &mut CommandEncoder) -> Result<Texture, image::ImageError> {
        let loaded_image = image::open(path)?;

        let rgba = loaded_image.into_rgba();
        let width = rgba.width();
        let height = rgba.height();

        let texture_extent = wgpu::Extent3d {
            width: width,
            height: height,
            depth: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(path.to_str().unwrap()),
            size: texture_extent,
            mip_level_count: 1,
            sample_count: 1,
            array_layer_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Bgra8Unorm,
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
        });

        let texture_view = texture.create_default_view();
        let raw = rgba.into_raw();

        let transfer_buffer = device.create_buffer_with_data(&raw, wgpu::BufferUsage::COPY_SRC);

        encoder.copy_buffer_to_texture(
            wgpu::BufferCopyView {
                buffer: &transfer_buffer,
                offset: 0,
                bytes_per_row: 4 * width * std::mem::size_of::<u8>() as u32,
                rows_per_image: height,
            },
            wgpu::TextureCopyView {
                texture: &texture,
                mip_level: 0,
                array_layer: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: 0,
                },
            },
            texture_extent,
        );

        Ok(Texture {
            buffer: texture,
            view: texture_view,
        })
    }
}

pub struct Document {
    pub textures: Vec<std::sync::Arc<Texture>>,
    pub brushes: crate::brush_editor::BrushData,
    pub paths: PathCollection,
}

impl Document {
    fn build(&self, builder: &mut lyon::path::Builder) {
        for path in self.paths.iter() {
            path.build(builder);
        }
    }

    fn hash(&self) -> u64 {
        let mut h = 0u64;
        for path in self.paths.iter() {
            h = h.wrapping_mul(32) ^ path.hash();
        }
        h
    }

    pub fn select_everything(&self) -> Selection {
        let mut selection = Selection { items: Vec::new() };
        for path in self.paths.iter() {
            for point in path.iter_points() {
                selection
                    .items
                    .push(VertexReference::new(path.path_index, point.index() as u32).into());
            }
        }
        selection
    }

    pub fn select_rect(&self, rect: CanvasRect) -> Selection {
        let mut selection = Selection { items: Vec::new() };
        for path in self.paths.iter() {
            for point in path.iter_points() {
                if rect.contains(point.position()) {
                    selection
                        .items
                        .push(VertexReference::new(path.path_index, point.index() as u32).into());
                }
            }
        }
        selection
    }
}



pub struct SceneParams {
    pub view: CanvasView,
    pub target_zoom: f32,
    pub target_scroll: CanvasVector,
    pub show_points: bool,
    pub show_wireframe: bool,
    pub stroke_width: f32,
    pub target_stroke_width: f32,
    pub draw_background: bool,
    pub cursor_position: (f32, f32),
    pub size_changed: bool,
}

fn update_inputs(event: Event<()>, control_flow: &mut ControlFlow, input: &mut InputManager, scene: &mut SceneParams, delta_time: f32) -> bool {
    let last_cursor = input.mouse_position;

    input.event(&event);

    match event {
        Event::MainEventsCleared => {
            return false;
        }
        Event::WindowEvent { event, .. } => match event {
            WindowEvent::Destroyed | WindowEvent::CloseRequested => {
                *control_flow = ControlFlow::Exit;
                return false;
            }
            WindowEvent::CursorMoved { position, .. } => {
                scene.cursor_position = (position.x as f32, position.y as f32);
                input.mouse_position = point(position.x as f32, position.y as f32);
            }
            WindowEvent::Resized(size) => {
                scene.view.resolution = size;
                scene.size_changed = true
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                scene.size_changed = true;
                println!("DPI changed");
            }
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        state,
                        virtual_keycode: Some(key),
                        ..
                    },
                ..
            } => {
                if state == ElementState::Pressed {
                    match key {
                        VirtualKeyCode::Escape => {
                            *control_flow = ControlFlow::Exit;
                            return false;
                        }
                        VirtualKeyCode::P => {
                            scene.show_points = !scene.show_points;
                        }
                        VirtualKeyCode::W => {
                            scene.show_wireframe = !scene.show_wireframe;
                        }
                        VirtualKeyCode::B => {
                            scene.draw_background = !scene.draw_background;
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        },
        Event::DeviceEvent {
            event: winit::event::DeviceEvent::Motion { axis: 3, value },
            ..
        } => {
            scene.target_zoom *= f32::powf(1.01, -value as f32);
        }
        _evt => {
            //println!("{:?}", _evt);
        }
    }

    if input.is_pressed(MouseButton::Right) {
        let cursor_delta = input.mouse_position - last_cursor;
        scene.target_scroll -= scene.view.screen_to_canvas_vector(cursor_delta);
        scene.view.scroll -= scene.view.screen_to_canvas_vector(cursor_delta);
    }

    if input.is_pressed(VirtualKeyCode::PageDown) {
        scene.target_zoom *= f32::powf(0.2, delta_time);
    }
    if input.is_pressed(VirtualKeyCode::PageUp) {
        scene.target_zoom *= f32::powf(5.0, delta_time);
    }
    if input.is_pressed(VirtualKeyCode::Left) {
        scene.target_scroll += scene.view.screen_to_canvas_vector(vector(-300.0, 0.0) * delta_time);
    }
    if input.is_pressed(VirtualKeyCode::Right) {
        scene.target_scroll += scene.view.screen_to_canvas_vector(vector(300.0, 0.0) * delta_time);
    }
    if input.is_pressed(VirtualKeyCode::Up) {
        scene.target_scroll += scene.view.screen_to_canvas_vector(vector(0.0, -300.0) * delta_time);
    }
    if input.is_pressed(VirtualKeyCode::Down) {
        scene.target_scroll += scene.view.screen_to_canvas_vector(vector(0.0, 300.0) * delta_time);
    }
    if input.is_pressed(VirtualKeyCode::A) {
        scene.target_stroke_width *= f32::powf(5.0, delta_time);
    }
    if input.is_pressed(VirtualKeyCode::Z) {
        scene.target_stroke_width *= f32::powf(0.2, delta_time);
    }

    //println!(" -- zoom: {}, scroll: {:?}", scene.target_zoom, scene.target_scroll);

    scene.view.zoom += (scene.target_zoom - scene.view.zoom) / 3.0;
    scene.view.scroll = scene.view.scroll + (scene.target_scroll - scene.view.scroll) / 3.0;
    scene.stroke_width = scene.stroke_width + (scene.target_stroke_width - scene.stroke_width) / 5.0;

    *control_flow = ControlFlow::Poll;

    return true;
}
