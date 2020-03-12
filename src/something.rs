use lyon::extra::rust_logo::build_logo_path;
use lyon::path::builder::*;
use lyon::path::Path;
use lyon::math::*;
use lyon::tessellation::geometry_builder::*;
use lyon::tessellation::basic_shapes::*;
use lyon::tessellation::{FillTessellator, FillOptions};
use lyon::tessellation::{StrokeTessellator, StrokeOptions};
use lyon::tessellation;

use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent, MouseButton};
use winit::event_loop::{EventLoop, ControlFlow};
use winit::window::Window;
use winit::dpi::{PhysicalSize};
use std::time::Instant;
use std::collections::{HashMap, HashSet};
use euclid;
use rand::{SeedableRng, rngs::StdRng};

use std::ops::Rem;
use crate::input::{InputManager, CircleShape, CapturedClick, KeyCombination};
use crate::fps_limiter::FPSLimiter;
use crate::geometry_utilities::types::*;
use crate::geometry_utilities::{poisson_disc_sampling, VectorField, VectorFieldPrimitive, sqr_distance_bezier_point};
use crate::path::*;
use crate::path_collection::{PathCollection, VertexReference, ControlPointReference, SelectionReference, ReferenceResolver, MutableReferenceResolver};
use crate::canvas::{CanvasView};

const PRIM_BUFFER_LEN: usize = 64;

#[repr(C)]
#[derive(Copy, Clone)]
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
        array_layer_count: 1,
        mip_level_count: 1,
        sample_count: sample_count,
        dimension: wgpu::TextureDimension::D2,
        format: sc_desc.format,
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
    };

    device.create_texture(multisampled_frame_descriptor).create_default_view()
}

fn load_shader(device: &wgpu::Device, shader_bytes : &[u8]) -> wgpu::ShaderModule {
    let spv = wgpu::read_spirv(std::io::Cursor::new(&shader_bytes)).unwrap();
    device.create_shader_module(&spv)
}


pub fn main() {
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

    let num_instances: u32 = PRIM_BUFFER_LEN as u32 - 1;
    let tolerance = 0.02;

    let t0 = Instant::now();
    // Build a Path for the rust logo.
    let mut builder = SvgPathBuilder::new(Path::builder());
    build_logo_path(&mut builder);
    let path = builder.build();

    let t1 = Instant::now();

    
    let mut geometry: VertexBuffers<GpuVertex, u32> = VertexBuffers::new();

    let stroke_prim_id = 0;
    let fill_prim_id = 1;

    let fill_count = FillTessellator::new().tessellate_path(
        &path,
        &FillOptions::tolerance(tolerance),
        &mut BuffersBuilder::new(&mut geometry, WithId(fill_prim_id as i32))
    ).unwrap();

    let t2 = Instant::now();

    StrokeTessellator::new().tessellate_path(
        &path,
        &StrokeOptions::tolerance(tolerance).dont_apply_line_width(),
        &mut BuffersBuilder::new(&mut geometry, WithId(stroke_prim_id as i32))
    ).unwrap();

    let t3 = Instant::now();

    let fill_range = 0..fill_count.indices;
    let stroke_range = fill_range.end..(geometry.indices.len() as u32);

    let mut bg_geometry: VertexBuffers<Point, u16> = VertexBuffers::new();
    fill_rectangle(
        &Rect::new(point(-1.0, -1.0), size(2.0, 2.0)),
        &FillOptions::default(),
        &mut BuffersBuilder::new(&mut bg_geometry, Positions),
    ).unwrap();

    let mut cpu_primitives = Vec::with_capacity(PRIM_BUFFER_LEN);
    for _ in 0..PRIM_BUFFER_LEN {
        cpu_primitives.push(
            Primitive {
                color: [1.0, 0.0, 0.0, 1.0],
                z_index: 0,
                width: 0.0,
                translate: [0.0, 0.0],
            },
        );
    }

    // Stroke primitive
    cpu_primitives[stroke_prim_id] = Primitive {
        color: [0.0, 0.0, 0.0, 0.5],
        z_index: num_instances as i32 + 2,
        width: 1.0,
        translate: [0.0, 0.0],
    };
    // Main fill primitive
    cpu_primitives[fill_prim_id] = Primitive {
        color: [1.0, 1.0, 1.0, 1.0],
        z_index: num_instances as i32 + 1,
        width: 0.0,
        translate: [0.0, 0.0],
    };
    // Instance primitives
    for idx in (fill_prim_id + 1)..(fill_prim_id + num_instances as usize) {
        cpu_primitives[idx].z_index = (idx as u32 + 1) as i32;
        cpu_primitives[idx].color = [
            (0.1 * idx as f32).rem(1.0),
            (0.5 * idx as f32).rem(1.0),
            (0.9 * idx as f32).rem(1.0),
            1.0,
        ];
    }

    let t4 = Instant::now();

    println!("Loading svg: {:?}", (t1.duration_since(t0).as_secs_f32()*1000.0));
    println!("Fill path: {:?}", (t2.duration_since(t1).as_secs_f32()*1000.0));
    println!("Stroke path: {:?}", (t3.duration_since(t2).as_secs_f32()*1000.0));
    println!("Memory: {:?}", (t4.duration_since(t3).as_secs_f32()*1000.0));

    let mut scene = SceneParams {
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
        path_editor: PathEditor::new(),
        input: InputManager::new()
    };

    let adapter = wgpu::Adapter::request(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        backends: wgpu::BackendBit::PRIMARY,
    }).unwrap();
    let (device, mut queue) = adapter.request_device(&wgpu::DeviceDescriptor {
        extensions: wgpu::Extensions {
            anisotropic_filtering: false,
        },
        limits: wgpu::Limits::default(),
    });

    let vbo = device
        .create_buffer_mapped(geometry.vertices.len(), wgpu::BufferUsage::VERTEX)
        .fill_from_slice(&geometry.vertices);

    let ibo = device
        .create_buffer_mapped(geometry.indices.len(), wgpu::BufferUsage::INDEX)
        .fill_from_slice(&geometry.indices);

    let bg_vbo = device
        .create_buffer_mapped(bg_geometry.vertices.len(), wgpu::BufferUsage::VERTEX)
        .fill_from_slice(&bg_geometry.vertices);

    let bg_ibo = device
        .create_buffer_mapped(bg_geometry.indices.len(), wgpu::BufferUsage::INDEX)
        .fill_from_slice(&bg_geometry.indices);

    let prim_buffer_byte_size = (PRIM_BUFFER_LEN * std::mem::size_of::<Primitive>()) as u64;
    let globals_buffer_byte_size = std::mem::size_of::<Globals>() as u64;

    let prims_ubo = device.create_buffer(
        &wgpu::BufferDescriptor {
            size: prim_buffer_byte_size,
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        }
    );

    let globals_ubo = device.create_buffer(
        &wgpu::BufferDescriptor {
            size: globals_buffer_byte_size,
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        }
    );

    let vs_module = load_shader(&device, include_bytes!("./../shaders/geometry.vert.spv"));
    let fs_module = load_shader(&device, include_bytes!("./../shaders/geometry.frag.spv"));
    let bg_vs_module = load_shader(&device, include_bytes!("./../shaders/background.vert.spv"));
    let bg_fs_module = load_shader(&device, include_bytes!("./../shaders/background.frag.spv"));

    let bind_group_layout = device.create_bind_group_layout(
        &wgpu::BindGroupLayoutDescriptor {
            bindings: &[
                wgpu::BindGroupLayoutBinding {
                    binding: 0,
                    visibility: wgpu::ShaderStage::VERTEX,
                    ty: wgpu::BindingType::UniformBuffer { dynamic: false },
                },
                wgpu::BindGroupLayoutBinding {
                    binding: 1,
                    visibility: wgpu::ShaderStage::VERTEX,
                    ty: wgpu::BindingType::UniformBuffer { dynamic: false },
                },
            ]
        }
    );
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        bindings: &[
            wgpu::Binding {
                binding: 0,
                resource: wgpu::BindingResource::Buffer {
                    buffer: &globals_ubo,
                    range: 0..globals_buffer_byte_size,
                },
            },
            wgpu::Binding {
                binding: 1,
                resource: wgpu::BindingResource::Buffer {
                    buffer: &prims_ubo,
                    range: 0..prim_buffer_byte_size,
                },
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
            color_blend: wgpu::BlendDescriptor { src_factor: wgpu::BlendFactor::SrcAlpha, dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha, operation: wgpu::BlendOperation::Add },
            alpha_blend: wgpu::BlendDescriptor { src_factor: wgpu::BlendFactor::SrcAlpha, dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha, operation: wgpu::BlendOperation::Add },
            write_mask: wgpu::ColorWrite::ALL,
        }],
        depth_stencil_state: depth_stencil_state.clone(),
        index_format: wgpu::IndexFormat::Uint32,
        vertex_buffers: &[
            wgpu::VertexBufferDescriptor {
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
                        format: wgpu::VertexFormat::Float,
                        shader_location: 2,
                    },
                ],
            },
        ],
        sample_count: sample_count,
        sample_mask: !0,
        alpha_to_coverage_enabled: false,
    };

    let render_pipeline = device.create_render_pipeline(&render_pipeline_descriptor);

    // TODO: this isn't what we want: we'd need the equivalent of VK_POLYGON_MODE_LINE,
    // but it doesn't seem to be exposed by wgpu?
    render_pipeline_descriptor.primitive_topology = wgpu::PrimitiveTopology::LineList;
    let wireframe_render_pipeline = device.create_render_pipeline(&render_pipeline_descriptor);

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
        index_format: wgpu::IndexFormat::Uint16,
        vertex_buffers: &[
            wgpu::VertexBufferDescriptor {
                stride: std::mem::size_of::<Point>() as u64,
                step_mode: wgpu::InputStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttributeDescriptor {
                        offset: 0,
                        format: wgpu::VertexFormat::Float2,
                        shader_location: 0,
                    },
                ],
            },
        ],
        sample_count: sample_count,
        sample_mask: !0,
        alpha_to_coverage_enabled: false,
    });

    let event_loop = EventLoop::new();
    let window = Window::new(&event_loop).unwrap();
    let size = window.inner_size();

    let mut swap_chain_desc = wgpu::SwapChainDescriptor {
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        format: wgpu::TextureFormat::Bgra8Unorm,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Vsync,
    };

    let mut multisampled_render_target = None;

    let window_surface = wgpu::Surface::create(&window);
    let mut swap_chain = device.create_swap_chain(
        &window_surface,
        &swap_chain_desc,
    );

    let mut depth_texture_view = None;

    let mut frame_count: f32 = 0.0;
    let mut last_time = Instant::now();
    let mut fps_limiter = FPSLimiter::new();

    event_loop.run(move |event, _, control_flow| {
        let new_time = Instant::now();
        let dt = (new_time.duration_since(last_time)).as_secs_f32();
        last_time = new_time;

        if update_inputs(event, control_flow, &mut scene, dt) {
            // keep polling inputs.
            return;
        }

        let t0 = Instant::now();
        scene.path_editor.update(&scene.view, &mut scene.input);
        dbg!(t0.elapsed());

        if scene.size_changed {
            scene.size_changed = false;
            let physical = scene.view.resolution;
            swap_chain_desc.width = physical.width;
            swap_chain_desc.height = physical.height;
            swap_chain = device.create_swap_chain(&window_surface, &swap_chain_desc);

            let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
                size: wgpu::Extent3d {
                    width: swap_chain_desc.width,
                    height: swap_chain_desc.height,
                    depth: 1,
                },
                array_layer_count: 1,
                mip_level_count: 1,
                sample_count: sample_count,
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

        let frame = swap_chain.get_next_texture();
        let mut a = Instant::now();
        let mut encoder = device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { todo: 0 }
        );

        cpu_primitives[stroke_prim_id as usize].width = scene.stroke_width;
        cpu_primitives[stroke_prim_id as usize].color = [
            (frame_count * 0.008 - 1.6).sin() * 0.1 + 0.1,
            (frame_count * 0.005 - 1.6).sin() * 0.1 + 0.1,
            (frame_count * 0.01 - 1.6).sin() * 0.1 + 0.1,
            0.5,
        ];

        for idx in 2..(num_instances+1) {
            cpu_primitives[idx as usize].translate = [
                (frame_count * 0.001 * idx as f32).sin() * (100.0 + idx as f32 * 10.0),
                (frame_count * 0.002 * idx as f32).sin() * (100.0 + idx as f32 * 10.0),
            ];
        }

        let globals_transfer_buffer = device.create_buffer_mapped(
            1,
            wgpu::BufferUsage::COPY_SRC,
        ).fill_from_slice(&[Globals {
            resolution: [scene.view.resolution.width as f32, scene.view.resolution.height as f32],
            zoom: scene.view.zoom,
            scroll_offset: scene.view.scroll.to_array(),
        }]);

        let prim_transfer_buffer = device.create_buffer_mapped(
            cpu_primitives.len(),
            wgpu::BufferUsage::COPY_SRC,
        );

        for (i, prim) in cpu_primitives.iter().enumerate() {
            prim_transfer_buffer.data[i] = *prim;
        }

        encoder.copy_buffer_to_buffer(
            &globals_transfer_buffer, 0,
            &globals_ubo, 0,
            globals_buffer_byte_size,
        );

        encoder.copy_buffer_to_buffer(
            &prim_transfer_buffer.finish(), 0,
            &prims_ubo, 0,
            prim_buffer_byte_size,
        );

        

        let mut test_geometry: VertexBuffers<GpuVertex, u32> = VertexBuffers::new();
        let mut builder = Path::builder();
        //builder.arc(Point::new(50.0 + 5.0, 50.0 + 0.0), vector(10.0, 5.0), Angle::degrees(300.0), Angle::degrees(1950 as f32));
        builder.line_to(point(100.0,0.0));
        builder.line_to(point(100.0,100.0));
        builder.line_to(point(0.0,100.0));
        builder.close();
        // *data.point_mut(0) = point(2.0, 3.0);
        // data.build(&mut builder);
        scene.path_editor.build(&mut builder);

        let p = builder.build();
        let mut canvas_tolerance : CanvasLength = ScreenLength::new(0.1) * scene.view.screen_to_canvas_scale();
        // It's important to clamp the tolerance to a not too small value
        // If the tesselator is fed a too small value it may get stuck in an infinite loop due to floating point precision errors
        canvas_tolerance = CanvasLength::new(canvas_tolerance.get().max(0.001));
        let canvas_line_width = ScreenLength::new(1.0) * scene.view.screen_to_canvas_scale();
        StrokeTessellator::new().tessellate_path(
            &p,
            &StrokeOptions::tolerance(canvas_tolerance.get()).with_line_width(canvas_line_width.get()),
            &mut BuffersBuilder::new(&mut test_geometry, WithId(0 as i32))
        ).unwrap();

        let test_vbo = device
        .create_buffer_mapped(test_geometry.vertices.len(), wgpu::BufferUsage::VERTEX)
        .fill_from_slice(&test_geometry.vertices);

        let indices : Vec<u32> = if scene.show_wireframe {
            // Transform the triangle primitives into line primitives: (0,1,2) => (0,1),(1,2),(2,0)
            test_geometry.indices.chunks_exact(3).flat_map(|v| vec![v[0], v[1], v[1], v[2], v[2], v[0]]).collect()
        } else {
            test_geometry.indices
        };
        dbg!(indices.len());
        
        let test_ibo = device
            .create_buffer_mapped(indices.len(), wgpu::BufferUsage::INDEX)
            .fill_from_slice(&indices);
    
        
        let test_transfer_buffer = device.create_buffer_mapped(
            1,
            wgpu::BufferUsage::COPY_SRC,
        );

        let test_ubo_size = (test_transfer_buffer.data.len() * std::mem::size_of::<Primitive>()) as u64;
        let test_ubo = device.create_buffer(
            &wgpu::BufferDescriptor {
                size: test_ubo_size,
                usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
            }
        );

        test_transfer_buffer.data[0] = Primitive {
            color: [1.0, 1.0, 1.0, 1.0],
            translate: [0.0, 0.0],
            z_index: 100,
            width: 0.0,
        };
        
        encoder.copy_buffer_to_buffer(
            &test_transfer_buffer.finish(), 0,
            &test_ubo, 0,
            test_ubo_size,
        );

        let test_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            bindings: &[
                wgpu::Binding {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer {
                        buffer: &globals_ubo,
                        range: 0..globals_buffer_byte_size,
                    },
                },
                wgpu::Binding {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer {
                        buffer: &test_ubo,
                        range: 0..test_ubo_size,
                    },
                },
            ],
        });
        

        {
            // A resolve target is only supported if the attachment actually uses anti-aliasing
            // So if sample_count == 1 then we must render directly to the swapchain's buffer
            let color_attachment = if let Some(msaa_target) = &multisampled_render_target {
                wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: msaa_target,
                    load_op: wgpu::LoadOp::Clear,
                    store_op: wgpu::StoreOp::Store,
                    clear_color: wgpu::Color::WHITE,
                    resolve_target: Some(&frame.view),
                }
            } else {
                wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: &frame.view,
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
                pass.set_index_buffer(&bg_ibo, 0);
                pass.set_vertex_buffers(0, &[(&bg_vbo, 0)]);

                pass.draw_indexed(0..6, 0, 0..1);
            }

            if scene.show_wireframe {
                pass.set_pipeline(&wireframe_render_pipeline);
            } else {
                pass.set_pipeline(&render_pipeline);
            }
            // pass.set_bind_group(0, &bind_group, &[]);
            // pass.set_index_buffer(&ibo, 0);
            // pass.set_vertex_buffers(0, &[(&vbo, 0)]);

            // pass.draw_indexed(fill_range.clone(), 0, 0..(num_instances as u32));
            // pass.draw_indexed(stroke_range.clone(), 0, 0..1);


            //pass.set_pipeline(&render_pipeline);
            pass.set_bind_group(0, &test_bind_group, &[]);
            pass.set_index_buffer(&test_ibo, 0);
            pass.set_vertex_buffers(0, &[(&test_vbo, 0)]);
            pass.draw_indexed(0..(indices.len() as u32), 0, 0..1);
        }

        queue.submit(&[encoder.finish()]);

        frame_count += 1.0;
        scene.input.tick_frame();
        dbg!(a.elapsed());
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

#[derive(Eq, Clone)]
struct Selection {
    /// Items in the selection.
    /// Should not contain any duplicates
    items: Vec<SelectionReference>,
}

impl PartialEq for Selection {
    fn eq(&self, other: &Self) -> bool {
        if self.items.len() != other.items.len() {
            return false;
        }

        let mut refs = HashSet::new();
        for item in &self.items {
            // Insert everything into the set
            // Also assert to make sure there are no duplicates in the selection
            // as this will break the algorithm
            assert!(refs.insert(item));
        }

        for item in &other.items {
            if !refs.contains(item) {
                return false
            }
        }

        // The selections have the same length and self contains everything in other.
        // This means they are identical since selections do not contain duplicates
        true
    }
}

struct ClosestInSelection {
    distance: CanvasLength,
    point: CanvasPoint,
    closest: ClosestItemInSelection,
}
enum ClosestItemInSelection {
    Curve { start: VertexReference },
    ControlPoint(ControlPointReference)
}

impl Selection {
    pub fn distance_to(&self, paths: &PathCollection, point: CanvasPoint) -> Option<ClosestInSelection> {
        let mut min_dist = std::f32::INFINITY;
        let mut closest_point = None;
        let mut closest_ref = None;
        let mut point_set = HashSet::new();
        for vertex in &self.items {
            if let SelectionReference::VertexReference(vertex) = vertex {
                point_set.insert(vertex);
            }
        }
        for vertex_ref in &self.items {
            match vertex_ref {
                SelectionReference::ControlPointReference(ctrl_ref) => {
                    let vertex = paths.resolve(ctrl_ref);
                    let dist = (vertex.position() - point).square_length();
                    if dist < min_dist {
                        min_dist = dist;
                        closest_point = Some(vertex.position());
                        closest_ref = Some(vertex_ref.clone());
                    }
                }
                SelectionReference::VertexReference(vertex_ref2) => {
                    let vertex = paths.resolve(vertex_ref2);
                    if vertex_ref2.next(paths).filter(|next| point_set.contains(&vertex_ref2)).is_some() {
                        let (dist, closest_on_curve) = sqr_distance_bezier_point(vertex.position(), vertex.control_after(), vertex.next().unwrap().control_before(), vertex.next().unwrap().position(), point);
                        if dist < min_dist {
                            min_dist = dist;
                            closest_point = Some(closest_on_curve);
                            closest_ref = Some(vertex_ref.clone());
                        }
                    } else {
                        let dist = (vertex.position() - point).square_length();
                        if dist < min_dist {
                            min_dist = dist;
                            closest_point = Some(vertex.position());
                            closest_ref = Some(vertex_ref.clone());
                        }
                    }
                }
            }
        }
        if let Some(p) = closest_ref {
            match p {
                SelectionReference::VertexReference(vertex_ref2) => {
                    Some(ClosestInSelection {
                        distance: CanvasLength::new(min_dist.sqrt()),
                        point: closest_point.unwrap(),
                        closest: ClosestItemInSelection::Curve { start: vertex_ref2 }
                    })
                }
                SelectionReference::ControlPointReference(ctrl_ref) => {
                    Some(ClosestInSelection {
                        distance: CanvasLength::new(min_dist.sqrt()),
                        point: closest_point.unwrap(),
                        closest: ClosestItemInSelection::ControlPoint(ctrl_ref)
                    })
                }
            }
        } else {
            None
        }
    }

    pub fn distance_to_curve(&self, paths: &PathCollection, point: CanvasPoint) -> Option<(VertexReference, CanvasPoint, CanvasLength)> {
        let mut min_dist = std::f32::INFINITY;
        let mut closest_point = None;
        let mut closest_ref = None;
        let mut point_set = HashSet::new();
        for vertex in &self.items {
            if let SelectionReference::VertexReference(vertex) = vertex {
                point_set.insert(vertex);
            }
        }
        for vertex_ref in &self.items {
            match vertex_ref {
                SelectionReference::VertexReference(vertex_ref2) => {
                    let vertex = paths.resolve(vertex_ref2);
                    if vertex_ref2.next(paths).filter(|next| point_set.contains(&vertex_ref2)).is_some() {
                        let (dist, closest_on_curve) = sqr_distance_bezier_point(vertex.position(), vertex.control_after(), vertex.next().unwrap().control_before(), vertex.next().unwrap().position(), point);
                        if dist < min_dist {
                            min_dist = dist;
                            closest_point = Some(closest_on_curve);
                            closest_ref = Some(vertex_ref2.clone());
                        }
                    } else {
                        let dist = (vertex.position() - point).square_length();
                        if dist < min_dist {
                            min_dist = dist;
                            closest_point = Some(vertex.position());
                            closest_ref = Some(vertex_ref2.clone());
                        }
                    }
                }
                _ => {}
            }
        }

        if let Some(closest_point) = closest_point {
            Some((closest_ref.unwrap(), closest_point, CanvasLength::new(min_dist.sqrt())))
        } else {
            None
        }
    }
}

fn smooth_vertices(selection: &Selection, paths: &mut PathCollection) {
    for vertex in &selection.items {
        if let SelectionReference::VertexReference(vertex) = vertex {
            let mut vertex = paths.resolve_mut(vertex);
            let pos = vertex.position();
            let prev = vertex.prev();
            let next = vertex.next();
            let dir = if prev.is_some() && next.is_some() {
                next.unwrap().position() - prev.unwrap().position()
            } else if let Some(prev) = prev {
                pos - prev.position()
            } else if let Some(next) = next {
                next.position() - pos
            } else {
                continue
            };
                
            vertex.set_control_before(pos - dir * 0.25);
            vertex.set_control_after(pos + dir * 0.25);
        }
    }
}

fn drag_at_point(selection: &Selection, paths: &PathCollection, point: CanvasPoint, distance_threshold: CanvasLength) -> Option<Selection> {
    let selected_vertices: Vec<VertexReference> = selection.items
        .iter()
        .filter_map(|v| if let SelectionReference::VertexReference(v_ref) = v { Some(*v_ref) } else { None })
        .collect();

    let potential_control_points = Selection {
        items: selected_vertices
            .iter()
            .flat_map(|v| vec![v.control_before(&paths).into(), v.control_after(&paths).into()])
            .collect()
    };

    // view.screen_to_canvas_point(capture.mouse_start)
    let distance_to_controls = potential_control_points.distance_to(&paths, point);
    let closest_vertex = selected_vertices.iter().map(|v| (v, (paths.resolve(v).position() - point).length())).min_by(|a,b| a.1.partial_cmp(&b.1).unwrap());
    let distance_to_curve = selection.distance_to_curve(&paths, point);
    
    // Figure out which item is the best to drag
    let mut best_score = CanvasLength::new(std::f32::INFINITY);
    // Lower weights means they are prioritized higher when picking
    const VERTEX_WEIGHT: f32 = 0.5;
    const CONTROL_WEIGHT: f32 = 0.7;
    const CURVE_WEIGHT: f32 = 1.0;

    if let Some(closest_vertex) = &closest_vertex {
        let dist = CanvasLength::new(closest_vertex.1);
        let score = dist * VERTEX_WEIGHT;
        if score < best_score && dist < distance_threshold {
            best_score = score;
        }
    }
    if let Some(distance_to_controls) = &distance_to_controls {
        let score = distance_to_controls.distance * CONTROL_WEIGHT;
        if score < best_score && distance_to_controls.distance < distance_threshold {
            best_score = score;
        }
    }
    if let Some(distance_to_curve) = &distance_to_curve {
        let score = distance_to_curve.2 * CURVE_WEIGHT;
        if score < best_score && distance_to_curve.2 < distance_threshold {
            best_score = score;
        }
    }

    // Figure out which item had the best score and then return that result
    if let Some(closest_vertex) = &closest_vertex {
        let dist = CanvasLength::new(closest_vertex.1);
        let score = dist * VERTEX_WEIGHT;
        if score == best_score {
            // The mouse started at a vertex, this means the user probably wants to drag the existing selection.
            return Some(selection.clone());
        }
    }

    if let Some(distance_to_controls) = &distance_to_controls {
        let score = distance_to_controls.distance * CONTROL_WEIGHT;
        if score == best_score {
            if let ClosestItemInSelection::ControlPoint(control_point) = distance_to_controls.closest {
                return Some(Selection { items: vec![control_point.into()] });
            } else {
                unreachable!();
            }
        }
    }

    if let Some(distance_to_curve) = &distance_to_curve {
        let score = distance_to_curve.2 * CURVE_WEIGHT;
        if score == best_score {
            // The mouse started at a selected curve, this means the user probably wants to drag the existing selection.
            return Some(selection.clone());
        }
    }
    
    // Nothing was close enough
    None
}

enum SelectState {
    Down(CapturedClick, Selection),
    DragSelect(CapturedClick),
    Dragging(CapturedClick, Selection, Vec<CanvasPoint>),
}



struct PathEditor {
    paths: PathCollection,
    ui_path: PathData,
    selected: Option<Selection>,
    select_state: Option<SelectState>,
    vector_field: VectorField,
}

impl PathEditor {
    fn new() -> PathEditor {
        PathEditor {
            paths: PathCollection { paths: vec![] },
            ui_path: PathData::new(),
            selected: None,
            select_state: None,
            vector_field: VectorField {
                primitives: vec![
                    VectorFieldPrimitive::Curl { center: point(0.0, 0.0), strength: 1.0, radius: 500.0 },
                    VectorFieldPrimitive::Curl { center: point(0.0, 50.0), strength: 1.0, radius: 500.0 },
                    VectorFieldPrimitive::Curl { center: point(100.0, 50.0), strength: 1.0, radius: 500.0 },
                    VectorFieldPrimitive::Curl { center: point(200.0, 300.0), strength: 1.0, radius: 2000.0 },
                    VectorFieldPrimitive::Linear { direction: vector(1.0, 1.0), strength: 1.1 },
                ]
            }
        }
    }

    fn update_ui(&mut self, view : &CanvasView, input: &InputManager) {
        let ui_path = &mut self.ui_path;
        ui_path.clear();
        let mouse_pos = view.screen_to_canvas_point(input.mouse_position);

        if let Some(selected) = &self.selected {
            for vertex in &selected.items {
                if let SelectionReference::VertexReference(vertex) = vertex {
                    let vertex = self.paths.resolve(vertex);
                    ui_path.add_circle(vertex.position(), ScreenLength::new(5.0) * view.screen_to_canvas_scale());

                    if vertex.control_before() != vertex.position() {
                        ui_path.add_circle(vertex.control_before(), ScreenLength::new(3.0) * view.screen_to_canvas_scale());
                    }
                    if vertex.control_after() != vertex.position() {
                        ui_path.add_circle(vertex.control_after(), ScreenLength::new(3.0) * view.screen_to_canvas_scale());
                    }

                    if vertex.control_before() != vertex.position() {
                        ui_path.move_to(vertex.control_before());
                        ui_path.line_to(vertex.position());
                    }
                    if vertex.control_after() != vertex.position() {
                        ui_path.move_to(vertex.control_after());
                        ui_path.line_to(vertex.position());
                    }
                }
            }

            let drag = drag_at_point(selected, &self.paths, mouse_pos, ScreenLength::new(5.0) * view.screen_to_canvas_scale());
            if let Some(drag) = drag {
                for item in &drag.items {
                    if let SelectionReference::VertexReference(vertex) = item {
                        let vertex = self.paths.resolve(vertex);
                        ui_path.add_circle(vertex.position(), ScreenLength::new(4.0) * view.screen_to_canvas_scale());
                    }
                    if let SelectionReference::ControlPointReference(vertex) = item {
                        let vertex = self.paths.resolve(vertex);
                        ui_path.add_circle(vertex.position(), ScreenLength::new(2.0) * view.screen_to_canvas_scale());
                    }
                }
            }
        }
        if let Some(SelectState::DragSelect(capture)) = &self.select_state {
            let start = view.screen_to_canvas_point(capture.mouse_start);
            let end = mouse_pos;
            ui_path.move_to(point(start.x, start.y));
            ui_path.line_to(point(end.x, start.y));
            ui_path.line_to(point(end.x, end.y));
            ui_path.line_to(point(start.x, end.y));
            ui_path.line_to(point(start.x, start.y));
            ui_path.close();
        }

        // let everything = self.select_everything();
        // if let Some((_, closest_point)) = everything.distance_to(mouse_pos) {
        //     dbg!(closest_point);
        //     ui_path.move_to(mouse_pos);
        //     ui_path.line_to(closest_point);
        //     ui_path.end();
        // }
        ui_path.add_circle(view.screen_to_canvas_point(input.mouse_position), CanvasLength::new(3.0));

        if self.paths.len() == 0 {
            self.paths.push(PathData::new());
        }

        let path = &mut self.paths.paths[0];
        if path.len() == 0 {
            path.clear();
            for p in &self.vector_field.primitives {
                match p {
                    &VectorFieldPrimitive::Curl { center, .. } => {
                        path.add_circle(center, CanvasLength::new(1.0));
                    }
                    &VectorFieldPrimitive::Linear { .. } => {

                    }
                }
            }

            let mut rng: StdRng = SeedableRng::seed_from_u64(0);
            let samples = poisson_disc_sampling(rect(-100.0, -100.0, 300.0, 300.0), 80.0, &mut rng);
            for (i, &p) in samples.iter().enumerate() {
                // if let VectorFieldPrimitive::Linear { ref mut strength, .. } = self.vector_field.primitives.last_mut().unwrap() {
                    // *strength = i as f32;
                // }
                path.move_to(p);
                let (vertices, closed) = self.vector_field.trace(p);
                for &p in vertices.iter().skip(1) {
                    path.line_to(p);
                }
                if closed {
                    path.close();
                } else {
                    path.end();
                }
            }
            // if let VectorFieldPrimitive::Linear { ref mut strength, .. } = self.vector_field.primitives.last_mut().unwrap() {
                // *strength = 2.0;
            // }
        }
        // let d = f32::sin(0.01 * (input.frame_count as f32));
        // for x in 0..100 {
        //     for y in 0..100 {
        //         let p = (point(x as f32, y as f32) - vector(50.0, 50.0)) * 4.0;
        //         let dir = self.vector_field.sample(p).unwrap();
        //         path.move_to(p);
        //         path.line_to(p + dir.normalize() * 5.0 * d);
        //         path.end();
        //     }
        // }
    }

    fn build(&self, builder: &mut lyon::path::Builder) {
        for path in self.paths.iter().chain(std::iter::once(&self.ui_path)) {
            path.build(builder);
        }
    }

    fn select_everything(&self) -> Selection {
        let mut selection = Selection {
            items: Vec::new()
        };
        for path in self.paths.iter() {
            for point in path.iter_points() {
                selection.items.push(VertexReference::new(path.path_index, point.index() as u32).into());
            }
        }
        selection
    }

    fn select_rect(&self, rect: CanvasRect) -> Selection {
        let mut selection = Selection {
            items: Vec::new()
        };
        for path in self.paths.iter() {
            for point in path.iter_points() {
                if rect.contains(point.position()) {
                    selection.items.push(VertexReference::new(path.path_index, point.index() as u32).into());
                }
            }
        }
        selection
    }

    fn update_selection(&mut self, view : &CanvasView, input: &mut InputManager) {
        self.select_state = match self.select_state.take() {
            None => {
                if let Some(mut capture) = input.capture_click_batch(winit::event::MouseButton::Left) {
                    for (path_index, path) in self.paths.iter().enumerate() {
                        for point in path.iter_points() {
                            let reference = VertexReference::new(path_index as u32, point.index() as u32);
                            capture.add(CircleShape { center: view.canvas_to_screen_point(point.position()), radius: 5.0 }, reference);
                        }
                    }
                    
                    if let Some((capture, best)) = capture.capture(input) {
                        // Something was close enough to the cursor, we should select it
                        Some(SelectState::Down(capture, Selection { items: vec![best.into()] }))
                    } else if let Some(capture) = input.capture_click(winit::event::MouseButton::Left) {
                        // Start with empty selection
                        Some(SelectState::Down(capture, Selection { items: vec![] }))
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            Some(state) => match state {
                // Note: !pressed matches should come first, otherwise mouse_up might be missed if multiple state transitions were to happen in a single frame
                SelectState::DragSelect(capture) if !capture.is_pressed(input) => {
                    // Only select if mouse was released inside the window
                    if capture.on_up(input) {
                        // Select
                        // TODO: Start point should be stored in canvas space, in case the view is zoomed or moved
                        let selection_rect = CanvasRect::from_points(vec![view.screen_to_canvas_point(capture.mouse_start), view.screen_to_canvas_point(input.mouse_position)]);
                        self.selected = Some(self.select_rect(selection_rect));
                        None
                    } else {
                        self.selected = None;
                        None
                    }
                }
                SelectState::Down(capture, selection) if !capture.is_pressed(input) => {
                    if capture.on_up(input) {
                        // Select
                        self.selected = Some(selection);
                    }
                    None
                }
                SelectState::Down(capture, _) if (capture.mouse_start - input.mouse_position).square_length() > 5.0*5.0 => {
                    if let Some(selection) = &self.selected {
                        let drag = drag_at_point(selection, &self.paths, view.screen_to_canvas_point(capture.mouse_start), ScreenLength::new(5.0) * view.screen_to_canvas_scale());

                        if let Some(drag) = drag {
                            let original_position = drag.items.iter().map(|x| x.position(&self.paths)).collect();
                            Some(SelectState::Dragging(capture, drag, original_position))
                        } else {
                            Some(SelectState::DragSelect(capture))
                        }
                    } else {
                        // Nothing selected. We should do a drag-select
                        Some(SelectState::DragSelect(capture))
                    }
                }
                SelectState::Dragging(capture, selection, original_positions) => {
                    let offset = view.screen_to_canvas_vector(input.mouse_position - capture.mouse_start);

                    // Move all points
                    for (vertex, &original_position) in selection.items.iter().zip(original_positions.iter()) {
                        if let SelectionReference::VertexReference(vertex) = vertex {
                            self.paths.resolve_mut(vertex).set_position(original_position + offset);
                        }
                    }

                    // Move all control points
                    // Doing this before moving points will lead to weird results
                    for (vertex, &original_position) in selection.items.iter().zip(original_positions.iter()) {
                        if let SelectionReference::ControlPointReference(vertex) = vertex {
                            // Move control point
                            self.paths.resolve_mut(vertex).set_position(original_position + offset);

                            // Move opposite control point
                            let center = self.paths.resolve(vertex).vertex().position();
                            let dir = original_position + offset - center;
                            self.paths.resolve_mut(&vertex.opposite_control(&self.paths)).set_position(center - dir);
                        }
                    }

                    if !capture.is_pressed(input) {
                        // Stop drag
                        None
                    } else {
                        Some(SelectState::Dragging(capture, selection, original_positions))
                    }
                }
                SelectState::DragSelect(..) => {
                    // Change visualization?
                    Some(state)
                }
                SelectState::Down(..) => Some(state)
            }
        };

        // self.select_state = state;
        // self.select_state.take 

        

        // if let SelectState::Down(capture, selection) = &self.select_state {
        //     if (capture.mouse_start - input.mouse_position).square_length() > 5.0*5.0 {
        //         self.select_state = SelectState::Dragging(*capture, *selection)
        //     }
        // }

        //     }
        //     SelectState::Down(capture, selection) => {
        //         if (capture.mouse_start - input.mouse_position).square_length() > 5.0*5.0 {
        //             SelectState::Dragging(capture, selection)
        //         } else {
        //             SelectState::Down(capture, selection)
        //         }
        //     }
        //     state => state
        // }
    }

    
    fn update(&mut self, view : &CanvasView, input: &mut InputManager) {
        let canvas_mouse_pos = view.screen_to_canvas_point(input.mouse_position);

        if input.is_pressed(VirtualKeyCode::T) {
            if let Some(_) = input.capture_click(winit::event::MouseButton::Left) {
                if self.paths.len() == 0 {
                    self.paths.push(PathData::new());
                }
                let path = self.paths.paths.last_mut().unwrap();
                path.line_to(canvas_mouse_pos);
            }
        }
        if input.on_combination(&KeyCombination::new().and(VirtualKeyCode::LControl).and(VirtualKeyCode::S)) {
            // Make smooth
            if let Some(selected) = &self.selected {
                smooth_vertices(selected, &mut self.paths);
            }
        }

        if input.is_pressed(VirtualKeyCode::A) {
            let everything = Some(self.select_everything());
            if self.selected == everything {
                self.selected = None;
            } else {
                self.selected = everything;
            }
        }

        self.update_selection(view, input);
        self.update_ui(view, input);
    }
}

struct SceneParams {
    view: CanvasView,
    target_zoom: f32,
    target_scroll: CanvasVector,
    show_points: bool,
    show_wireframe: bool,
    stroke_width: f32,
    target_stroke_width: f32,
    draw_background: bool,
    cursor_position: (f32, f32),
    size_changed: bool,
    input: InputManager,
    path_editor: PathEditor,
}

fn update_inputs(event: Event<()>, control_flow: &mut ControlFlow, scene: &mut SceneParams, delta_time: f32) -> bool {
    let last_cursor = scene.input.mouse_position;

    scene.input.event(&event);

    match event {
        Event::MainEventsCleared => {
            return false;
        }
        Event::WindowEvent { event, .. } => {
            match event {
                WindowEvent::Destroyed | WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                    return false;
                }
                WindowEvent::CursorMoved { position, ..} => {
                    scene.cursor_position = (position.x as f32, position.y as f32);
                    scene.input.mouse_position = point(position.x as f32, position.y as f32);
                }
                WindowEvent::Resized(size) => {
                    scene.view.resolution = size;
                    scene.size_changed = true
                }
                WindowEvent::ScaleFactorChanged {..} => {
                    scene.size_changed = true;
                    println!("DPI changed");
                }
                WindowEvent::KeyboardInput {
                    input: KeyboardInput {
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
            }
        }
        Event::DeviceEvent { event: winit::event::DeviceEvent::Motion { axis: 3, value }, ..} => {
            scene.target_zoom *= f32::powf(1.01, -value as f32);
        }
        _evt => {
            //println!("{:?}", _evt);
        }
    }

    if scene.input.is_pressed(MouseButton::Right) {
        let cursor_delta = scene.input.mouse_position - last_cursor;
        scene.target_scroll -= scene.view.screen_to_canvas_vector(cursor_delta);
        scene.view.scroll -= scene.view.screen_to_canvas_vector(cursor_delta);
    }

    if scene.input.is_pressed(VirtualKeyCode::PageDown) {
        scene.target_zoom *= f32::powf(0.2, delta_time);
    }
    if scene.input.is_pressed(VirtualKeyCode::PageUp) {
        scene.target_zoom *= f32::powf(5.0, delta_time);
    }
    if scene.input.is_pressed(VirtualKeyCode::Left) {
        scene.target_scroll += scene.view.screen_to_canvas_vector(vector(-300.0, 0.0) * delta_time);
    }
    if scene.input.is_pressed(VirtualKeyCode::Right) {
        scene.target_scroll += scene.view.screen_to_canvas_vector(vector(300.0, 0.0) * delta_time);
    }
    if scene.input.is_pressed(VirtualKeyCode::Up) {
        scene.target_scroll -= scene.view.screen_to_canvas_vector(vector(0.0, -300.0) * delta_time);
    }
    if scene.input.is_pressed(VirtualKeyCode::Down) {
        scene.target_scroll += scene.view.screen_to_canvas_vector(vector(0.0, 300.0) * delta_time);
    }
    if scene.input.is_pressed(VirtualKeyCode::A) {
        scene.target_stroke_width *= f32::powf(5.0, delta_time);
    }
    if scene.input.is_pressed(VirtualKeyCode::Z) {
        scene.target_stroke_width *= f32::powf(0.2, delta_time);
    }

    //println!(" -- zoom: {}, scroll: {:?}", scene.target_zoom, scene.target_scroll);

    scene.view.zoom += (scene.target_zoom - scene.view.zoom) / 3.0;
    scene.view.scroll = scene.view.scroll + (scene.target_scroll - scene.view.scroll) / 3.0;
    scene.stroke_width = scene.stroke_width +
        (scene.target_stroke_width - scene.stroke_width) / 5.0;

    *control_flow = ControlFlow::Poll;

    return true;
}
