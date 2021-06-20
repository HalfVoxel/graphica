use euclid::point2 as point;
use euclid::rect;
use euclid::size2 as size;
use euclid::vec2 as vector;
use euclid::Size2D;
use lyon::math::Point;
use lyon::path::Path;
use lyon::tessellation::geometry_builder::*;
use lyon::tessellation::FillOptions;
use lyon::tessellation::{StrokeOptions, StrokeTessellator};
use std::num::NonZeroU64;

use wgpu::Extent3d;

use wgpu::TextureView;
use wgpu_glyph::{ab_glyph::FontArc, GlyphBrushBuilder};

use std::rc::Rc;
use std::time::Instant;
use wgpu::{
    util::StagingBelt, BindGroup, Buffer, CommandEncoder, CommandEncoderDescriptor, Device, RenderPipeline,
    TextureDescriptor,
};
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, Event, KeyboardInput, MouseButton, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::Window;

use crate::blitter::BlitOp;
use crate::brush_manager::{BrushGpuVertex, BrushManager, CloneBrushGpuVertex};
use crate::cache::ephermal_buffer_cache::EphermalBufferCache;
use crate::cache::material_cache::MaterialCache;
use crate::cache::render_pipeline_cache::RenderPipelineCache;
use crate::cache::render_texture_cache::RenderTextureCache;
use crate::canvas::CanvasView;
use crate::egui_wrapper::EguiWrapper;
use crate::encoder::Encoder;
use crate::fps_limiter::FPSLimiter;
use crate::geometry_utilities;
use crate::geometry_utilities::types::*;
use crate::geometry_utilities::ParamCurveDistanceEval;
use crate::gui;
use crate::input::{InputManager, KeyCombination};
use crate::path::*;
use crate::path_collection::{PathCollection, VertexReference};
use crate::path_editor::*;

use crate::brush_editor::{BrushData, BrushEditor};
use crate::toolbar::GUIRoot;
use crate::{blitter::Blitter, vertex::GPUVertex};
use crate::{
    texture::{RenderTexture, SwapchainImageWrapper, Texture},
    wgpu_utils::*,
};
use arrayvec::ArrayVec;
use async_std::task;

#[cfg(feature = "profile")]
use cpuprofiler::PROFILER;

use cgmath::Matrix4;
use kurbo::CubicBez;
use kurbo::Point as KurboPoint;
use palette::Pixel;
use palette::Srgba;
use std::sync::Arc;

#[repr(C)]
#[derive(Copy, Clone, Default)]
struct Globals {
    resolution: [f32; 2],
    scroll_offset: [f32; 2],
    zoom: f32,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct PosNormVertex {
    position: [f32; 2],
    normal: [f32; 2],
    prim_id: i32,
}

impl GPUVertex for PosNormVertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as u64,
            step_mode: wgpu::InputStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    format: wgpu::VertexFormat::Float32x2,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    offset: 8,
                    format: wgpu::VertexFormat::Float32x2,
                    shader_location: 1,
                },
                wgpu::VertexAttribute {
                    offset: 16,
                    format: wgpu::VertexFormat::Sint32,
                    shader_location: 2,
                },
            ],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
struct Primitive {
    mvp_matrix: Matrix4<f32>,
    color: [f32; 4],
    width: f32,
}

const DEFAULT_WINDOW_WIDTH: u32 = 2048;
const DEFAULT_WINDOW_HEIGHT: u32 = 2048;

/// Creates a texture that uses MSAA and fits a given swap chain
fn create_multisampled_framebuffer(
    device: &wgpu::Device,
    size: &wgpu::Extent3d,
    sample_count: u32,
    format: wgpu::TextureFormat,
) -> Texture {
    Texture::new(
        device,
        wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            // array_layer_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format, //sc_desc.format,
            usage: wgpu::TextureUsage::RENDER_ATTACHMENT,
            label: Some("MSAA Framebuffer"),
        },
    )
}

struct DocumentRenderer {
    vbo: Buffer,
    ibo: Buffer,
    scene_ubo: Buffer,
    #[allow(dead_code)]
    primitive_ubo: Buffer,
    bind_group: BindGroup,
    index_buffer_length: usize,
    render_pipeline: Rc<RenderPipeline>,
    brush_renderer: BrushRenderer,
}

pub struct BrushRenderer {
    vbo: Buffer,
    ibo: Buffer,
    // ubo: Buffer,
    index_buffer_length: usize,
    bind_group: BindGroup,
    brush_manager: Rc<BrushManager>,
    stroke_ranges: Vec<std::ops::Range<u32>>,
}

fn sample_points_along_curve(path: &PathData, spacing: f32) -> Vec<(usize, CanvasPoint)> {
    let mut result = vec![];

    for sub_path in path.iter_sub_paths() {
        sample_points_along_sub_path(&sub_path, spacing, &mut result);
    }

    result
}

fn sample_points_along_sub_path(sub_path: &SubPath, spacing: f32, result: &mut Vec<(usize, CanvasPoint)>) {
    let mut offset = 0f32;

    for start in sub_path.iter_beziers() {
        let end = start.next().unwrap();
        let p0 = start.position();
        let p1 = start.control_after();
        let p2 = end.control_before();
        let p3 = end.position();
        let bezier = CubicBez::new(
            KurboPoint::new(p0.x as f64, p0.y as f64),
            KurboPoint::new(p1.x as f64, p1.y as f64),
            KurboPoint::new(p2.x as f64, p2.y as f64),
            KurboPoint::new(p3.x as f64, p3.y as f64),
        );

        loop {
            match bezier.eval_at_distance((spacing + offset) as f64, 0.01) {
                Ok(p) => {
                    result.push((start.index, CanvasPoint::new(p.x as f32, p.y as f32)));
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

#[derive(Copy, Clone)]
#[allow(dead_code)]
struct BrushUniforms {
    mvp_matrix: Matrix4<f32>,
}

pub struct BrushRendererWithReadback {
    ibo: Buffer,
    vbo: Buffer,
    brush_manager: Rc<BrushManager>,
    temp_texture_view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    points: Vec<(usize, CanvasPoint)>,
    size: f32,
    brush_texture: Arc<Texture>,
}

impl BrushRendererWithReadback {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        brush_data: &BrushData,
        view: &CanvasView,
        device: &Device,
        _encoder: &mut CommandEncoder,
        _scene_ubo: &Buffer,
        _scene_ubo_size: u64,
        brush_manager: &Rc<BrushManager>,
        texture: &Arc<Texture>,
    ) -> BrushRendererWithReadback {
        let points = sample_points_along_curve(&brush_data.path, 1.41);

        let to_normalized_pos = |v: CanvasPoint| view.screen_to_normalized(view.canvas_to_screen_point(v));

        let size = 20.0;
        let color = Srgba::new(1.0, 1.0, 1.0, 1.0).into_format().into_raw();
        let vertices: Vec<CloneBrushGpuVertex> = points
            .windows(2)
            .flat_map(|window| {
                let prev = window[0].1;
                let pos = window[1].1;
                let clone_pos = prev;
                ArrayVec::from([
                    CloneBrushGpuVertex {
                        position: pos + vector(-size, -size),
                        uv_background_source: to_normalized_pos(clone_pos + vector(-size, -size)),
                        uv_background_target: to_normalized_pos(pos + vector(-size, -size)),
                        uv_brush: point(0.0, 0.0),
                        color,
                    },
                    CloneBrushGpuVertex {
                        position: pos + vector(size, -size),
                        uv_background_source: to_normalized_pos(clone_pos + vector(size, -size)),
                        uv_background_target: to_normalized_pos(pos + vector(size, -size)),
                        uv_brush: point(1.0, 0.0),
                        color,
                    },
                    CloneBrushGpuVertex {
                        position: pos + vector(size, size),
                        uv_background_source: to_normalized_pos(clone_pos + vector(size, size)),
                        uv_background_target: to_normalized_pos(pos + vector(size, size)),
                        uv_brush: point(1.0, 1.0),
                        color,
                    },
                    CloneBrushGpuVertex {
                        position: pos + vector(-size, size),
                        uv_background_source: to_normalized_pos(clone_pos + vector(-size, size)),
                        uv_background_target: to_normalized_pos(pos + vector(-size, size)),
                        uv_brush: point(0.0, 1.0),
                        color,
                    },
                ])
            })
            .collect();

        let (vbo, _) = create_buffer(device, &vertices, wgpu::BufferUsage::VERTEX, None);

        #[allow(clippy::identity_op)]
        let indices: Vec<u32> = (0..points.len() as u32)
            .flat_map(|x| vec![4 * x + 0, 4 * x + 1, 4 * x + 2, 4 * x + 3, 4 * x + 2, 4 * x + 0])
            .collect();
        let (ibo, _) = create_buffer(device, &indices, wgpu::BufferUsage::INDEX, None);

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            lod_min_clamp: -100.0,
            lod_max_clamp: 100.0,
            compare: None,
            anisotropy_clamp: None,
            border_color: None,
            label: None,
        });

        let width_in_pixels = (2.0 * (CanvasLength::new(size) * view.canvas_to_screen_scale()).get()).round() as u32;

        let texture_extent = wgpu::Extent3d {
            width: width_in_pixels,
            height: width_in_pixels,
            depth_or_array_layers: 1,
        };

        let temp_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Temp texture"),
            size: texture_extent,
            mip_level_count: 1,
            sample_count: 1,
            // array_layer_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: crate::config::TEXTURE_FORMAT,
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::RENDER_ATTACHMENT,
        });

        let temp_texture_view = temp_texture.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            brush_manager: brush_manager.clone(),
            temp_texture_view,
            vbo,
            ibo,
            sampler,
            points,
            size,
            brush_texture: texture.clone(),
        }
    }

    pub fn render(&self, encoder: &mut Encoder, view: &CanvasView) {
        if self.points.len() <= 1 {
            return;
        }

        let temp_to_frame_blitter =
            encoder
                .blitter
                .with_textures(encoder.device, &self.temp_texture_view, encoder.target_texture.view);

        let bind_group = encoder.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.brush_manager.splat_with_readback.bind_group_layout,
            label: Some("Clone brush Bind Group"),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(encoder.target_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.brush_texture.view),
                },
            ],
        });

        let to_normalized_pos = |v: CanvasPoint| view.screen_to_normalized(view.canvas_to_screen_point(v));

        for (mut i, (_, p)) in self.points.iter().enumerate() {
            // First point is a noop
            if i == 0 {
                continue;
            }
            i -= 1;

            // First pass
            {
                let mut pass = encoder.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Brush with readback. First pass."),
                    color_attachments: &[wgpu::RenderPassColorAttachment {
                        view: &self.temp_texture_view,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::RED),
                            store: true,
                        },
                        resolve_target: None,
                    }],
                    depth_stencil_attachment: None,
                });

                pass.set_pipeline(&self.brush_manager.splat_with_readback.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.set_index_buffer(self.ibo.slice(..), wgpu::IndexFormat::Uint32);
                pass.set_vertex_buffer(0, self.vbo.slice(..));
                pass.draw_indexed((i * 6) as u32..((i + 1) * 6) as u32, 0, 0..1);
            }

            // Second pass, copy back
            {
                let mn = to_normalized_pos(*p - vector(self.size, self.size));
                let mx = to_normalized_pos(*p + vector(self.size, self.size));
                let r = rect(mn.x, mn.y, mx.x - mn.x, mx.y - mn.y);
                // let uv = vertices[(i*4)..(i+1)*4].iter().map(|x| x.uv_background_target).collect::<ArrayVec<[Point;4]>>();
                // let r = Rect::from_points(uv);
                temp_to_frame_blitter.blit(encoder.device, encoder.encoder, rect(0.0, 0.0, 1.0, 1.0), r, 1, None);
            }
        }

        encoder.blitter.blit(
            encoder.device,
            encoder.encoder,
            encoder.target_texture.view,
            encoder.multisampled_render_target.as_ref().unwrap().view,
            rect(0.0, 0.0, 1.0, 1.0),
            rect(0.0, 0.0, 1.0, 1.0),
            8,
            None,
        );
    }
}

pub struct BrushRendererWithReadbackBatched {
    ubo: Buffer,
    ubo_size: u64,
    brush_manager: Rc<BrushManager>,
    size_in_pixels: u32,
    temp_texture_view: wgpu::TextureView,
    points: Vec<(usize, CanvasPoint)>,
    brush_texture: Arc<Texture>,
    sampler: wgpu::Sampler,
}

#[repr(C, align(16))]
#[derive(Copy, Clone)]
struct ReadbackPrimitive {
    origin_src: (i32, i32),
    origin_dst: (i32, i32),
}

#[repr(C, align(16))]
struct ReadbackUniforms {
    width_per_group: i32,
    height_per_group: i32,
    num_primitives: i32,
}

impl BrushRendererWithReadbackBatched {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        brush_data: &BrushData,
        _view: &CanvasView,
        device: &Device,
        _encoder: &mut CommandEncoder,
        _scene_ubo: &Buffer,
        _scene_ubo_size: u64,
        brush_manager: &Rc<BrushManager>,
        texture: &Arc<Texture>,
    ) -> BrushRendererWithReadbackBatched {
        let points = sample_points_along_curve(&brush_data.path, 1.41);

        let size_in_pixels = 32;
        let offset = -vector(size_in_pixels as f32 * 0.5, size_in_pixels as f32 * 0.5);
        let primitives: Vec<ReadbackPrimitive> = points
            .windows(2)
            .map(|window| {
                let clone_pos = window[0].1 + offset;
                let pos = window[1].1 + offset;

                ReadbackPrimitive {
                    origin_src: (clone_pos.x.round() as i32, clone_pos.y.round() as i32),
                    origin_dst: (pos.x.round() as i32, pos.y.round() as i32),
                }
            })
            .collect();

        let (ubo, ubo_size) = create_buffer(device, &primitives, wgpu::BufferUsage::STORAGE, None);

        const LOCAL_SIZE: u32 = 32;
        let width_per_group = (size_in_pixels + LOCAL_SIZE - 1) / LOCAL_SIZE;

        let texture_extent = wgpu::Extent3d {
            width: width_per_group * 32,
            height: width_per_group * 32,
            depth_or_array_layers: 1,
        };

        let temp_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Temp texture"),
            size: texture_extent,
            mip_level_count: 1,
            sample_count: 1,
            // array_layer_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: crate::config::TEXTURE_FORMAT,
            usage: wgpu::TextureUsage::STORAGE,
        });

        let temp_texture_view = temp_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: None,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            lod_min_clamp: -100.0,
            lod_max_clamp: 100.0,
            compare: None,
            anisotropy_clamp: None,
            border_color: None,
        });

        // let width_in_pixels = (2.0 * (CanvasLength::new(size) * view.canvas_to_screen_scale()).get()).round() as u32;

        Self {
            brush_manager: brush_manager.clone(),
            ubo,
            ubo_size,
            size_in_pixels,
            temp_texture_view,
            sampler,
            points,
            brush_texture: texture.clone(),
        }
    }

    pub fn render(&self, encoder: &mut Encoder, _view: &CanvasView) {
        if self.ubo_size == 0 {
            return;
        }

        const LOCAL_SIZE: u32 = 32;

        let width_per_group = (self.size_in_pixels + LOCAL_SIZE - 1) / LOCAL_SIZE;
        let height_per_group = (self.size_in_pixels + LOCAL_SIZE - 1) / LOCAL_SIZE;

        let (settings_ubo, settings_ubo_size) = create_buffer(
            encoder.device,
            &[ReadbackUniforms {
                width_per_group: width_per_group as i32,
                height_per_group: height_per_group as i32,
                num_primitives: self.points.len() as i32 - 1,
            }],
            wgpu::BufferUsage::UNIFORM,
            "Readback settings",
        );

        let bind_group = encoder.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.brush_manager.splat_with_readback_batched.bind_group_layout,
            label: Some("Clone brush Bind Group"),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(encoder.target_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.temp_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.brush_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.ubo,
                        offset: 0,
                        // TODO: Use None?
                        size: Some(NonZeroU64::new(self.ubo_size).unwrap()),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &settings_ubo,
                        offset: 0,
                        size: Some(NonZeroU64::new(settings_ubo_size).unwrap()),
                    }),
                },
            ],
        });

        let mut cpass = encoder.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("brush with readback"),
        });
        cpass.set_pipeline(&self.brush_manager.splat_with_readback_batched.pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch(1, 1, 1);
    }
}

impl BrushRenderer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        brush_data: &BrushData,
        view: &CanvasView,
        device: &Device,
        _encoder: &mut CommandEncoder,
        scene_ubo: &Buffer,
        scene_ubo_size: u64,
        brush_manager: &Rc<BrushManager>,
        texture: &Arc<Texture>,
    ) -> BrushRenderer {
        let size = brush_data.brush.size;
        let mut vertices: Vec<BrushGpuVertex> = vec![];
        let mut stroke_ranges = vec![];

        for sub_path in brush_data.path.iter_sub_paths() {
            // let mut points = vec![];
            let start_vertex = vertices.len();
            let subpath_points = sub_path
                .iter_points()
                .map(|p| p.position())
                .collect::<Vec<CanvasPoint>>();
            let smoothed_subpath = crate::catmull_rom::catmull_rom_smooth(&subpath_points);

            // sample_points_along_sub_path(&sub_path, brush_data.brush.spacing * size, &mut points);
            let points = sample_points_along_curve(&smoothed_subpath, brush_data.brush.spacing * size);

            vertices.extend(points.iter().flat_map(|&(vertex_index, pos)| {
                let color = brush_data.colors[vertex_index / 3].into_format().into_raw();
                ArrayVec::from([
                    BrushGpuVertex {
                        position: pos + vector(-size, -size),
                        uv: point(0.0, 0.0),
                        color,
                    },
                    BrushGpuVertex {
                        position: pos + vector(size, -size),
                        uv: point(1.0, 0.0),
                        color,
                    },
                    BrushGpuVertex {
                        position: pos + vector(size, size),
                        uv: point(1.0, 1.0),
                        color,
                    },
                    BrushGpuVertex {
                        position: pos + vector(-size, size),
                        uv: point(0.0, 1.0),
                        color,
                    },
                ])
            }));

            let start_triangle = ((start_vertex / 4) * 6) as u32;
            let end_triangle = ((vertices.len() / 4) * 6) as u32;
            stroke_ranges.push(start_triangle..end_triangle);
        }

        let (vbo, _) = create_buffer(device, &vertices, wgpu::BufferUsage::VERTEX, None);

        #[allow(clippy::identity_op)]
        let indices: Vec<u32> = (0..(vertices.len() / 4) as u32)
            .flat_map(|x| vec![4 * x + 0, 4 * x + 1, 4 * x + 2, 4 * x + 3, 4 * x + 2, 4 * x + 0])
            .collect();
        let (ibo, _) = create_buffer(device, &indices, wgpu::BufferUsage::INDEX, None);

        let view_matrix = view.canvas_to_view_matrix();

        let (primitive_ubo, primitive_ubo_size) = create_buffer(
            device,
            &[BrushUniforms {
                mvp_matrix: view_matrix * Matrix4::from_translation([0.0, 0.0, 0.1].into()),
            }],
            wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
            "Brush Primitive UBO",
        );

        // let primitive_ubo_transfer = device.create_buffer_with_data(as_u8_slice(&[BrushUniforms { dummy: 0 }]), wgpu::BufferUsage::COPY_SRC);
        // let primitive_ubo_size = std::mem::size_of::<BrushUniforms>() as u64;
        // let primitive_ubo = device.create_buffer(&wgpu::BufferDescriptor {
        //     size: primitive_ubo_size,
        //     usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        // });
        // encoder.copy_buffer_to_buffer(&primitive_ubo_transfer, 0, &primitive_ubo, 0, primitive_ubo_size);

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Brush sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            lod_min_clamp: -100.0,
            lod_max_clamp: 100.0,
            compare: None,
            anisotropy_clamp: None,
            border_color: None,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &brush_manager.splat.bind_group_layout,
            label: Some("Brush bind group"),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: scene_ubo,
                        offset: 0,
                        size: Some(NonZeroU64::new(scene_ubo_size).unwrap()),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &primitive_ubo,
                        offset: 0,
                        size: Some(NonZeroU64::new(primitive_ubo_size).unwrap()),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
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
            brush_manager: brush_manager.clone(),
            stroke_ranges,
        }
    }

    pub fn update(&mut self, _view: &CanvasView, _device: &Device, _encoder: &mut CommandEncoder) {
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

    pub fn render(&self, encoder: &mut Encoder, _view: &CanvasView) {
        if self.index_buffer_length == 0 {
            return;
        }

        // let blitter = encoder.blitter.with_textures(encoder.device, &encoder.scratch_texture.view, encoder.target_texture);
        for stroke_range in &self.stroke_ranges {
            {
                let mut pass = encoder.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("brush"),
                    color_attachments: &[wgpu::RenderPassColorAttachment {
                        view: &encoder.scratch_texture.view,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: true,
                        },
                        resolve_target: None,
                    }],
                    depth_stencil_attachment: None,
                });

                pass.set_pipeline(&self.brush_manager.splat.pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.set_index_buffer(self.ibo.slice(..), wgpu::IndexFormat::Uint32);
                pass.set_vertex_buffer(0, self.vbo.slice(..));
                pass.draw_indexed(stroke_range.clone(), 0, 0..1);
            }

            encoder.blitter.blend(
                encoder.device,
                encoder.encoder,
                &encoder.scratch_texture.view,
                encoder.target_texture.view,
                (encoder.resolution.width, encoder.resolution.height),
            );
        }
    }
}

impl DocumentRenderer {
    #[allow(clippy::too_many_arguments)]
    fn new(
        document: &Document,
        view: &CanvasView,
        device: &Device,
        encoder: &mut CommandEncoder,
        staging_belt: &mut StagingBelt,
        bind_group_layout: &wgpu::BindGroupLayout,
        wireframe: bool,
        render_pipeline: &Rc<RenderPipeline>,
        wireframe_render_pipeline: &Rc<RenderPipeline>,
        brush_manager: &Rc<BrushManager>,
    ) -> DocumentRenderer {
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

        let (vbo, _) = create_buffer(device, &geometry.vertices, wgpu::BufferUsage::VERTEX, "Document VBO");
        let (ibo, _) = create_buffer(device, &geometry.indices, wgpu::BufferUsage::INDEX, "Document IBO");

        let (scene_ubo, scene_ubo_size) = create_buffer(
            device,
            &[Globals {
                resolution: [view.resolution.width as f32, view.resolution.height as f32],
                zoom: view.zoom,
                scroll_offset: view.scroll.to_array(),
            }],
            wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
            "Document UBO",
        );

        let view_matrix = view.canvas_to_view_matrix();

        let (primitive_ubo, primitive_ubo_size) = create_buffer(
            device,
            &[Primitive {
                color: [1.0, 1.0, 1.0, 1.0],
                mvp_matrix: view_matrix * Matrix4::from_translation([0.0, 0.0, 0.1].into()),
                width: 0.0,
            }],
            wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
            "Document Primitive UBO",
        );

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: bind_group_layout,
            label: Some("Document bind group"),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &scene_ubo,
                        offset: 0,
                        size: Some(NonZeroU64::new(scene_ubo_size).unwrap()),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &primitive_ubo,
                        offset: 0,
                        size: Some(NonZeroU64::new(primitive_ubo_size).unwrap()),
                    }),
                },
            ],
        });

        let brush_renderer = BrushRenderer::new(
            &document.brushes,
            view,
            device,
            encoder,
            &scene_ubo,
            scene_ubo_size,
            brush_manager,
            &document.textures[0],
        );

        let mut res = DocumentRenderer {
            vbo,
            ibo,
            scene_ubo,
            primitive_ubo,
            bind_group,
            index_buffer_length: geometry.indices.len(),
            brush_renderer,
            render_pipeline: if !wireframe {
                render_pipeline.clone()
            } else {
                wireframe_render_pipeline.clone()
            },
        };
        res.update(view, device, encoder, staging_belt);
        res
    }

    fn update(
        &mut self,
        view: &CanvasView,
        device: &Device,
        encoder: &mut CommandEncoder,
        staging_belt: &mut StagingBelt,
    ) {
        // TODO: Verify expected size?
        update_buffer_via_transfer(
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

    fn render(&self, encoder: &mut Encoder, view: &CanvasView) {
        puffin::profile_function!();
        {
            let mut pass = encoder.begin_msaa_render_pass(None, Some("document render pass"));
            pass.set_pipeline(&self.render_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_index_buffer(self.ibo.slice(..), wgpu::IndexFormat::Uint32);
            pass.set_vertex_buffer(0, self.vbo.slice(..));
            pass.draw_indexed(0..(self.index_buffer_length as u32), 0, 0..1);
        }

        self.brush_renderer.render(encoder, view);
    }
}

fn render_background(
    encoder: &mut Encoder,
    bg_pipeline: &RenderPipeline,
    bind_group: &BindGroup,
    bg_ibo: &Buffer,
    bg_vbo: &Buffer,
) {
    puffin::profile_function!();
    let mut pass = encoder.begin_msaa_render_pass(Some(wgpu::Color::RED), Some("background render pass"));
    pass.set_pipeline(bg_pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.set_index_buffer(bg_ibo.slice(..), wgpu::IndexFormat::Uint16);
    pass.set_vertex_buffer(0, bg_vbo.slice(..));

    pass.draw_indexed(0..6, 0, 0..1);
}

fn blit_document_to_window<'a>(
    hl_encoder: &mut Encoder<'a>,
    scene: &SceneParams,
    doc_size: Size2D<u32, CanvasSpace>,
    window_extent: Extent3d,
    source: &TextureView,
    target: &RenderTexture,
) -> BlitOp<'a> {
    puffin::profile_function!();
    let canvas_in_screen_space =
        scene
            .view
            .canvas_to_screen_rect(rect(0.0, 0.0, doc_size.width as f32, doc_size.height as f32));
    let canvas_in_screen_uv_space =
        canvas_in_screen_space.scale(1.0 / window_extent.width as f32, 1.0 / window_extent.height as f32);

    let blittex = hl_encoder
        .blitter
        .with_textures(hl_encoder.device, source, target.default_view().view);

    blittex.blit_regions(
        hl_encoder.device,
        rect(0.0, 0.0, 1.0, 1.0),
        canvas_in_screen_uv_space.to_untyped(),
        target.sample_count(),
    )
}

#[allow(unused_variables)]
pub fn main() {
    #[cfg(feature = "profile")]
    PROFILER
        .lock()
        .unwrap()
        .start("./my-prof.profile")
        .expect("Couldn't start");

    println!("== wgpu example ==");
    println!("Controls:");
    println!("  Arrow keys: scrolling");
    println!("  PgUp/PgDown: zoom in/out");
    println!("  w: toggle wireframe mode");
    println!("  b: toggle drawing the background");
    println!("  a/z: increase/decrease the stroke width");
    puffin::set_scopes_on(true);

    let mut data = PathData::default();
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
    let mut tessellator = lyon::tessellation::FillTessellator::new();
    tessellator
        .tessellate_rectangle(
            &euclid::Rect::new(point(-1.0 * 5.0, -1.0 * 5.0), size(2.0 * 5.0, 2.0 * 5.0)),
            &FillOptions::default(),
            &mut BuffersBuilder::new(&mut bg_geometry, Positions),
        )
        .unwrap();

    let t4 = Instant::now();

    println!("Loading svg: {:?}", (t1.duration_since(t0).as_secs_f32() * 1000.0));
    println!("Memory: {:?}", (t4.duration_since(t3).as_secs_f32() * 1000.0));

    let document = Document {
        size: Some(size(2048, 2048)),
        paths: PathCollection { paths: vec![] },
        brushes: crate::brush_editor::BrushData::new(),
        textures: vec![],
    };
    let mut ui_document = Document {
        size: None,
        paths: PathCollection { paths: vec![] },
        brushes: crate::brush_editor::BrushData::new(),
        textures: vec![],
    };

    let mut gui = gui::Root::default();
    let gui_root = gui.add(GUIRoot::new());

    let mut editor = Editor {
        path_editor: PathEditor::new(&mut ui_document),
        brush_editor: BrushEditor::new(),
        document,
        ui_document,
        gui,
        gui_root,
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
        input: InputManager::default(),
    };

    let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);
    let adapter = task::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None, // TODO
    }))
    .unwrap();

    let (device, queue) = task::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: None,
            features: wgpu::Features::NON_FILL_POLYGON_MODE | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
            limits: wgpu::Limits::default(),
        },
        None,
    ))
    .expect("Failed to request device");

    let mut staging_belt = wgpu::util::StagingBelt::new(1024);

    let vs_module = crate::shader::load_wgsl_shader(&device, "shaders/geometry.wgsl");
    let bg_module = crate::shader::load_wgsl_shader(&device, "shaders/background.wgsl");
    // let vs_module = load_shader(&device, "shaders/geometry.vert.spv");
    // let fs_module = load_shader(&device, "shaders/geometry.frag.spv");
    // let bg_vs_module = load_shader(&device, "shaders/background.vert.spv");
    // let bg_fs_module = load_shader(&device, "shaders/background.frag.spv");

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Geometry Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStage::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStage::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        push_constant_ranges: &[],
        bind_group_layouts: &[&bind_group_layout],
    });

    let depth_stencil_state = Some(wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth32Float,
        depth_write_enabled: true,
        depth_compare: wgpu::CompareFunction::Greater,
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    });

    let mut render_pipeline_descriptor = wgpu::RenderPipelineDescriptor {
        label: Some("Main Render Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &vs_module,
            entry_point: "vs_main",
            buffers: &[PosNormVertex::desc()],
        },
        fragment: Some(wgpu::FragmentState {
            module: &vs_module,
            entry_point: "fs_main",
            targets: &[wgpu::ColorTargetState {
                format: crate::config::TEXTURE_FORMAT,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrite::ALL,
            }],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            polygon_mode: wgpu::PolygonMode::Fill,
            clamp_depth: false,
            conservative: false,
        },
        depth_stencil: depth_stencil_state.clone(),
        multisample: wgpu::MultisampleState {
            count: sample_count,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
    };

    let render_pipeline = Rc::new(device.create_render_pipeline(&render_pipeline_descriptor));
    render_pipeline_descriptor.primitive.polygon_mode = wgpu::PolygonMode::Line;
    let wireframe_render_pipeline = Rc::new(device.create_render_pipeline(&render_pipeline_descriptor));

    let bg_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Background pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &bg_module,
            entry_point: "vs_main",
            buffers: &[Point::desc()],
        },
        fragment: Some(wgpu::FragmentState {
            module: &bg_module,
            entry_point: "fs_main",
            targets: &[wgpu::ColorTargetState {
                format: crate::config::TEXTURE_FORMAT,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrite::ALL,
            }],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            polygon_mode: wgpu::PolygonMode::Fill,
            clamp_depth: false,
            conservative: false,
        },
        depth_stencil: depth_stencil_state,
        multisample: wgpu::MultisampleState {
            count: sample_count,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
    });

    let brush_manager = Rc::new(BrushManager::load(&device, sample_count));

    let event_loop = EventLoop::new();
    let window = Window::new(&event_loop).unwrap();
    window.set_inner_size(PhysicalSize::new(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT));
    let size = window.inner_size();

    let mut swap_chain_desc = wgpu::SwapChainDescriptor {
        usage: wgpu::TextureUsage::RENDER_ATTACHMENT,
        format: crate::config::FRAMEBUFFER_TEXTURE_FORMAT,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Fifo,
    };

    let mut multisampled_render_target = None;
    let multisampled_render_target_document;

    let window_surface = unsafe { instance.create_surface(&window) };
    let mut swap_chain = device.create_swap_chain(&window_surface, &swap_chain_desc);

    let mut depth_texture = None;
    let depth_texture_document;

    {
        let document_extent = wgpu::Extent3d {
            width: editor.document.size.unwrap().width,
            height: editor.document.size.unwrap().height,
            depth_or_array_layers: 1,
        };
        depth_texture_document = Some(Rc::new(Texture::new(
            &device,
            wgpu::TextureDescriptor {
                label: Some("Framebuffer depth"),
                size: document_extent,
                mip_level_count: 1,
                sample_count,
                // array_layer_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsage::RENDER_ATTACHMENT,
            },
        )));

        multisampled_render_target_document = if sample_count > 1 {
            Some(Rc::new(create_multisampled_framebuffer(
                &device,
                &document_extent,
                sample_count,
                crate::config::TEXTURE_FORMAT,
            )))
        } else {
            None
        };
    }

    let mut frame_count: f32 = 0.0;
    let mut last_time = Instant::now();
    let mut fps_limiter = FPSLimiter::default();
    let mut last_hash1 = 0u64;
    let mut last_hash2 = 0u64;
    let mut document_renderer1: Option<DocumentRenderer> = None;
    let mut document_renderer2: Option<DocumentRenderer> = None;

    let mut init_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Init encoder"),
    });

    let blitter = Blitter::new(&device, &mut init_encoder);

    let (bg_vbo, _) = create_buffer(&device, &bg_geometry.vertices, wgpu::BufferUsage::VERTEX, "BG VBO");
    let (bg_ibo, _) = create_buffer(&device, &bg_geometry.indices, wgpu::BufferUsage::INDEX, "BG IBO");

    let bg_ubo_data = &[Primitive {
        color: [1.0, 1.0, 1.0, 1.0],
        mvp_matrix: Matrix4::from_translation([0.0, 0.0, 100.0].into()),
        width: 0.0,
    }];
    let (bg_ubo, bg_ubo_size) = create_buffer(&device, bg_ubo_data, wgpu::BufferUsage::UNIFORM, "BG UBO");

    let (globals_ubo, globals_ubo_size) = create_buffer(
        &device,
        &[Globals { ..Default::default() }],
        wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        "Globals UBO",
    );

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        label: Some("Background Bind Group"),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &globals_ubo,
                    offset: 0,
                    size: Some(NonZeroU64::new(globals_ubo_size).unwrap()),
                }),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &bg_ubo,
                    offset: 0,
                    size: Some(NonZeroU64::new(bg_ubo_size).unwrap()),
                }),
            },
        ],
    });

    let _blur = crate::blur::BlurCompute::new(&device);

    let tex = std::sync::Arc::new(
        Texture::load_from_file(std::path::Path::new("brush.png"), &device, &mut init_encoder).unwrap(),
    );
    editor.document.textures.push(tex.clone());
    editor.ui_document.textures.push(tex);

    queue.submit(std::iter::once(init_encoder.finish()));

    let mipmapper = crate::mipmap::Mipmapper::new(&device);

    let font: Vec<u8> = std::fs::read("fonts/Bitter-Regular.ttf").expect("Could not find font");
    let font = FontArc::try_from_vec(font).unwrap();
    let glyph_brush = GlyphBrushBuilder::using_font(font).build(&device, crate::config::TEXTURE_FORMAT);

    let document_extent = wgpu::Extent3d {
        width: editor.document.size.unwrap().width,
        height: editor.document.size.unwrap().height,
        depth_or_array_layers: 1,
    };

    // Render into this texture
    let temp_document_frame = Rc::new(Texture::new(
        &device,
        wgpu::TextureDescriptor {
            label: Some("Temp frame texture"),
            size: document_extent,
            mip_level_count: crate::mipmap::max_mipmaps(document_extent),
            sample_count: 1,
            // array_layer_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: crate::config::TEXTURE_FORMAT,
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::RENDER_ATTACHMENT | wgpu::TextureUsage::STORAGE,
        },
    ));

    let scratch_texture = Arc::new(Texture::new(
        &device,
        wgpu::TextureDescriptor {
            label: Some("Scratch texture"),
            size: document_extent,
            mip_level_count: 1,
            sample_count: 1,
            // array_layer_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: crate::config::TEXTURE_FORMAT,
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::RENDER_ATTACHMENT | wgpu::TextureUsage::STORAGE,
        },
    ));

    let start_time = Instant::now();

    let mut egui_wrapper = EguiWrapper::new(&device, size, window.scale_factor());
    let mut render_pipeline_cache = RenderPipelineCache::default();
    let mut ephermal_buffer_cache = EphermalBufferCache::default();
    let mut render_texture_cache = RenderTextureCache::default();
    let mut material_cache = MaterialCache::default();

    event_loop.run(move |event, _, control_flow| {
        let scene = &mut editor.scene;
        {
            puffin::profile_scope!("event handling");
            egui_wrapper.platform.handle_event(&event);

            let new_time = Instant::now();
            let dt = (new_time.duration_since(last_time)).as_secs_f32();
            last_time = new_time;

            if update_inputs(event, control_flow, &mut editor.input, scene, dt) {
                // keep polling inputs.
                return;
            }
        }

        puffin::GlobalProfiler::lock().new_frame();
        puffin::profile_scope!("event loop");

        egui_wrapper.platform.update_time(start_time.elapsed().as_secs_f64());

        editor.gui.update();
        editor.gui.input(&mut editor.ui_document, &mut editor.input);
        editor.gui.render(&mut editor.ui_document, &scene.view);
        // editor.toolbar.update_ui(&mut editor.ui_document, &mut editor.document, &scene.view, &mut editor.input);
        editor.path_editor.update(
            &mut editor.ui_document,
            &mut editor.document,
            &scene.view,
            &mut editor.input,
            &editor.gui_root.get(&editor.gui).tool,
        );
        editor.brush_editor.update(
            &mut editor.ui_document,
            &mut editor.document,
            &scene.view,
            &mut editor.input,
            &editor.gui_root.get(&editor.gui).tool,
        );

        if editor.input.on_combination(
            &KeyCombination::new()
                .and(VirtualKeyCode::LControl)
                .and(VirtualKeyCode::W),
        ) {
            // Quit
            *control_flow = ControlFlow::Exit;

            #[cfg(feature = "profile")]
            PROFILER.lock().unwrap().stop().expect("Couldn't stop");
            return;
        }

        // println!("Path editor = {:?}", t0.elapsed());

        if scene.size_changed {
            let physical = scene.view.resolution;
            swap_chain_desc.width = physical.width;
            swap_chain_desc.height = physical.height;
        }

        let window_extent = wgpu::Extent3d {
            width: swap_chain_desc.width,
            height: swap_chain_desc.height,
            depth_or_array_layers: 1,
        };

        if scene.size_changed {
            puffin::profile_scope!("rebuild swapchain");
            println!("Rebuilding swap chain");
            scene.size_changed = false;
            swap_chain = device.create_swap_chain(&window_surface, &swap_chain_desc);
            depth_texture = Some(Rc::new(Texture::new(
                &device,
                TextureDescriptor {
                    label: Some("Framebuffer depth"),
                    size: window_extent,
                    // array_layer_count: 1,
                    mip_level_count: 1,
                    sample_count,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Depth32Float,
                    usage: wgpu::TextureUsage::RENDER_ATTACHMENT,
                },
            )));

            multisampled_render_target = if sample_count > 1 {
                Some(Rc::new(create_multisampled_framebuffer(
                    &device,
                    &window_extent,
                    sample_count,
                    swap_chain_desc.format,
                )))
            } else {
                None
            };
        }

        let swapchain_output = {
            puffin::profile_scope!("aquire swapchain");
            swap_chain.get_current_frame().unwrap()
        };
        let frame = RenderTexture::from(SwapchainImageWrapper::from_swapchain_image(
            swapchain_output,
            &swap_chain_desc,
        ));
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Frame encoder"),
        });

        let doc_size = editor.document.size.unwrap();
        let dummy_view = CanvasView {
            zoom: 1.0,
            scroll: vector(0.0, 0.0),
            resolution: PhysicalSize::new(doc_size.width, doc_size.height),
        };

        scene.update_uniform_globals(&device, &mut encoder, &mut staging_belt, &globals_ubo);

        let hash = editor.document.hash() ^ scene.view.hash();
        if hash != last_hash1 || document_renderer1.is_none() {
            last_hash1 = hash;
            document_renderer1 = Some(DocumentRenderer::new(
                &editor.document,
                &dummy_view,
                &device,
                &mut encoder,
                &mut staging_belt,
                &bind_group_layout,
                scene.show_wireframe,
                &render_pipeline,
                &wireframe_render_pipeline,
                &brush_manager,
            ));
        }

        let ui_view = CanvasView {
            zoom: 1.0,
            scroll: vector(0.0, 0.0),
            resolution: scene.view.resolution,
        };
        let hash = editor.ui_document.hash() ^ ui_view.hash();
        let bypass_hash_check = true;
        if hash != last_hash2 || document_renderer2.is_none() || bypass_hash_check {
            last_hash2 = hash;
            document_renderer2 = Some(DocumentRenderer::new(
                &editor.ui_document,
                &ui_view,
                &device,
                &mut encoder,
                &mut staging_belt,
                &bind_group_layout,
                scene.show_wireframe,
                &render_pipeline,
                &wireframe_render_pipeline,
                &brush_manager,
            ));
        }

        // {
        let msaa_render_target = RenderTexture::from(multisampled_render_target.as_ref().unwrap().clone());
        let depth_target = RenderTexture::from(depth_texture.as_ref().unwrap().clone());
        let mut hl_encoder = Encoder {
            device: &device,
            encoder: &mut encoder,
            multisampled_render_target: Some(msaa_render_target.default_view()),
            target_texture: frame.default_view(),
            depth_texture_view: depth_target.default_view(),
            blitter: &blitter,
            resolution: document_extent,
            scratch_texture: scratch_texture.clone(),
        };

        render_background(&mut hl_encoder, &bg_pipeline, &bind_group, &bg_ibo, &bg_vbo);

        document_renderer1
            .as_mut()
            .unwrap()
            .update(&dummy_view, &device, &mut encoder, &mut staging_belt);
        document_renderer2
            .as_mut()
            .unwrap()
            .update(&ui_view, &device, &mut encoder, &mut staging_belt);

        {
            {
                let msaa_render_target =
                    RenderTexture::from(multisampled_render_target_document.as_ref().unwrap().clone());
                let render_target = RenderTexture::from(temp_document_frame.clone());
                let depth_target = RenderTexture::from(depth_texture_document.as_ref().unwrap().clone());
                let mut hl_encoder = Encoder {
                    device: &device,
                    encoder: &mut encoder,
                    multisampled_render_target: Some(msaa_render_target.default_view()),
                    target_texture: render_target.get_mip_level_view(0).unwrap(),
                    depth_texture_view: depth_target.default_view(),
                    blitter: &blitter,
                    resolution: document_extent,
                    scratch_texture: scratch_texture.clone(),
                };
                // render_main_view(&scene);

                if scene.draw_background {
                    render_background(&mut hl_encoder, &bg_pipeline, &bind_group, &bg_ibo, &bg_vbo);
                } else {
                    hl_encoder.begin_msaa_render_pass(Some(wgpu::Color::RED), Some("background clear pass"));
                }

                document_renderer1
                    .as_ref()
                    .unwrap()
                    .render(&mut hl_encoder, &dummy_view);
            }

            blitter.rgb_to_srgb(
                &device,
                &mut encoder,
                &temp_document_frame.view,
                (document_extent.width, document_extent.height),
            );
            mipmapper.generate_mipmaps(&device, &mut encoder, &temp_document_frame);

            let msaa_render_target = multisampled_render_target.clone().map(RenderTexture::from);
            let depth_target = RenderTexture::from(depth_texture.as_ref().unwrap().clone());
            let mut hl_encoder = Encoder {
                device: &device,
                encoder: &mut encoder,
                multisampled_render_target: msaa_render_target.as_ref().map(RenderTexture::default_view),
                target_texture: frame.default_view(),
                depth_texture_view: depth_target.default_view(),
                blitter: &blitter,
                resolution: document_extent,
                scratch_texture: scratch_texture.clone(),
            };

            let blitop = blit_document_to_window(
                &mut hl_encoder,
                scene,
                doc_size,
                window_extent,
                &temp_document_frame.view,
                msaa_render_target.as_ref().unwrap_or(&frame),
            );

            // {
            //     // Clear pass
            //     let mut pass = hl_encoder.begin_msaa_render_pass(
            //         Some(wgpu::Color {
            //             r: 120.0 / 255.0,
            //             g: 41.0 / 255.0,
            //             b: 41.0 / 255.0,
            //             a: 1.0,
            //         }),
            //         Some("clear pass"),
            //     );

            //     blitop.render(&mut pass);
            // }

            // document_renderer2
            //     .as_ref()
            //     .unwrap()
            //     .render(&mut hl_encoder, &scene.view);

            ephermal_buffer_cache.reset();

            let canvas_in_screen_space =
            scene
                .view
                .canvas_to_screen_rect(rect(0.0, 0.0, doc_size.width as f32, doc_size.height as f32));
        let canvas_in_screen_uv_space =
            canvas_in_screen_space.scale(1.0 / window_extent.width as f32, 1.0 / window_extent.height as f32);

            let mut render_graph = crate::render_graph::RenderGraph::default();
            let mut t = render_graph.clear(Size2D::new(frame.size().width, frame.size().height), wgpu::Color::GREEN);
            let blue = render_graph.clear(Size2D::new(128, 128), wgpu::Color::BLUE);
            t = render_graph.blit(blue, t, rect(0.0, 0.0, 128.0, 128.0), canvas_in_screen_space.cast_unit());
            let mut render_graph_compiler = crate::render_graph::RenderGraphCompiler {
                device: &device,
                encoder: &mut encoder,
                blitter: &blitter,
                render_pipeline_cache: &mut render_pipeline_cache,
                ephermal_buffer_cache: &mut ephermal_buffer_cache,
                render_texture_cache: &mut render_texture_cache,
                material_cache: &mut material_cache,
                staging_belt: &mut staging_belt,
            };
            let passes = render_graph_compiler.compile(&render_graph, t, &frame);
            render_graph_compiler.render(&passes);

            // // // blur.render(&mut hl_encoder);

            // let section = Section {
            //     screen_position: (10.0, 10.0),
            //     text: vec![Text::new("Hello wgpu_gfilyphåäöЎaњ").with_scale(36.0)],
            //     ..Section::default() // color, position, etc
            // };

            // glyph_brush.queue(section);

            // glyph_brush
            //     .draw_queued(
            //         &device,
            //         &mut staging_belt,
            //         &mut encoder,
            //         frame.default_view().view,
            //         window_extent.width,
            //         window_extent.height,
            //     )
            //     .unwrap();
        }

        // blitter.blit(
        //     &device,
        //     &mut encoder,
        //     &temp_window_frame_view,
        //     frame.default_view(),
        //     rect(0.0, 0.0, 1.0, 1.0),
        //     rect(0.0, 0.0, 1.0, 1.0),
        //     1,
        // );

        staging_belt.finish();

        // Upload all resources for the GPU.
        let screen_descriptor = egui_wgpu_backend::ScreenDescriptor {
            physical_width: window_extent.width,
            physical_height: window_extent.height,
            scale_factor: window.scale_factor() as f32,
        };

        let paint_jobs = egui_wrapper.frame(|ctx| {
            egui::SidePanel::left("side", 300.0).show(&ctx, |ui| {
                ui.label("Hello");
                if ui.button("Hello").clicked() {
                    ui.label("BLAH!");
                }
            });

            puffin_egui::profiler_window(&ctx);
        });

        egui_wrapper.render(
            &paint_jobs,
            &device,
            frame.default_view().view,
            &queue,
            &mut encoder,
            &screen_descriptor,
        );

        {
            puffin::profile_scope!("queue.submit");
            queue.submit(std::iter::once(encoder.finish()));
        }
        let (sender, mut receiver) = futures::channel::oneshot::channel();
        let recall = staging_belt.recall();
        task::spawn(async move {
            recall.await;
            let _ = sender.send(());
        });

        // dbg!(t3.elapsed());

        frame_count += 1.0;
        editor.input.tick_frame();
        // println!("Preparing GPU work = {:?}", t1.elapsed());
        fps_limiter.wait(std::time::Duration::from_secs_f32(1.0 / 60.0));
        {
            puffin::profile_scope!("wait for device");
            while receiver.try_recv().is_err() {
                device.poll(wgpu::Maintain::Wait);
            }
        }
    });
}

/// This vertex constructor forwards the positions and normals provided by the
/// tessellators and add a shape id.
pub struct WithId(pub i32);

impl FillVertexConstructor<PosNormVertex> for WithId {
    fn new_vertex(&mut self, vertex: lyon::tessellation::FillVertex) -> PosNormVertex {
        debug_assert!(!vertex.position().x.is_nan());
        debug_assert!(!vertex.position().y.is_nan());
        PosNormVertex {
            position: vertex.position().to_array(),
            normal: [0.0, 0.0],
            prim_id: self.0,
        }
    }
}

impl StrokeVertexConstructor<PosNormVertex> for WithId {
    fn new_vertex(&mut self, vertex: lyon::tessellation::StrokeVertex) -> PosNormVertex {
        debug_assert!(!vertex.position().x.is_nan());
        debug_assert!(!vertex.position().y.is_nan());
        debug_assert!(!vertex.normal().x.is_nan());
        debug_assert!(!vertex.normal().y.is_nan());
        debug_assert!(!vertex.advancement().is_nan());
        PosNormVertex {
            position: vertex.position().to_array(),
            normal: vertex.normal().to_array(),
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

pub struct Document {
    pub size: Option<euclid::Size2D<u32, CanvasSpace>>,
    pub textures: Vec<std::sync::Arc<Texture>>,
    pub brushes: crate::brush_editor::BrushData,
    pub paths: PathCollection,
}

impl Document {
    fn build(&self, builder: &mut lyon::path::path::Builder) {
        for path in self.paths.iter() {
            path.build(builder);
        }
    }

    fn hash(&self) -> u64 {
        puffin::profile_function!();
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

impl SceneParams {
    pub fn update_uniform_globals(
        &self,
        device: &Device,
        encoder: &mut CommandEncoder,
        staging_belt: &mut StagingBelt,
        globals_ubo: &Buffer,
    ) {
        update_buffer_via_transfer(
            device,
            encoder,
            staging_belt,
            &[Globals {
                resolution: [self.view.resolution.width as f32, self.view.resolution.height as f32],
                zoom: self.view.zoom,
                scroll_offset: self.view.scroll.to_array(),
            }],
            globals_ubo,
        );
    }
}

fn update_inputs(
    event: Event<()>,
    control_flow: &mut ControlFlow,
    input: &mut InputManager,
    scene: &mut SceneParams,
    delta_time: f32,
) -> bool {
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
        scene.view.scroll += scene.view.screen_to_canvas_vector(vector(-300.0, 0.0) * delta_time);
    }
    if input.is_pressed(VirtualKeyCode::Right) {
        scene.view.scroll += scene.view.screen_to_canvas_vector(vector(300.0, 0.0) * delta_time);
    }
    if input.is_pressed(VirtualKeyCode::Up) {
        scene.view.scroll += scene.view.screen_to_canvas_vector(vector(0.0, -300.0) * delta_time);
    }
    if input.is_pressed(VirtualKeyCode::Down) {
        scene.view.scroll += scene.view.screen_to_canvas_vector(vector(0.0, 300.0) * delta_time);
    }
    if input.is_pressed(VirtualKeyCode::A) {
        scene.target_stroke_width *= f32::powf(5.0, delta_time);
    }
    if input.is_pressed(VirtualKeyCode::Z) {
        scene.target_stroke_width *= f32::powf(0.2, delta_time);
    }

    //println!(" -- zoom: {}, scroll: {:?}", scene.target_zoom, scene.target_scroll);
    let new_zoom = scene.view.zoom + (scene.target_zoom - scene.view.zoom) / 3.0;
    scene.view.zoom_around_point(input.mouse_position, new_zoom);
    scene.stroke_width = scene.stroke_width + (scene.target_stroke_width - scene.stroke_width) / 5.0;

    *control_flow = ControlFlow::Poll;

    true
}
