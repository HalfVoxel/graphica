use euclid::point2 as point;
use euclid::rect;
use euclid::size2 as size;
use euclid::vec2 as vector;
use euclid::Size2D;

use image::ColorType;
use lyon::math::Point;

use lyon::tessellation::geometry_builder::*;
use lyon::tessellation::FillOptions;

use rand::Rng;
use std::convert::TryInto;
use std::num::NonZeroU64;
use std::ops::Range;
use std::sync::Mutex;
use wgpu::BlendState;
use wgpu::CompositeAlphaMode;

use wgpu::Extent3d;

// use wgpu_glyph::{ab_glyph::FontArc, GlyphBrushBuilder};

use crate::brush_manager::{BrushGpuVertex, BrushManager, CloneBrushGpuVertex};
use crate::cache::ephermal_buffer_cache::BufferRange;
use crate::cache::ephermal_buffer_cache::EphermalBufferCache;
use crate::cache::material_cache::BindGroupEntryArc;
use crate::cache::material_cache::BindingResourceArc;
use crate::cache::material_cache::DynamicMaterial;
use crate::cache::material_cache::Material;
use crate::cache::material_cache::MaterialCache;
use crate::cache::render_pipeline_cache::RenderPipelineBase;
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
use crate::input::CapturedDrag;
use crate::input::{InputManager, KeyCombination};
use crate::nodes::vector_strokes::Primitive;
use crate::nodes::BlitNode;
use crate::nodes::Cache;
use crate::nodes::Clear;
use crate::nodes::VectorRenderer;
use crate::path::*;
use crate::path_collection::{PathCollection, VertexReference};
use crate::path_editor::*;
use crate::persistent_graph::PersistentGraph;
use crate::persistent_graph::RenderNode;
use crate::render_graph::GraphNode;
use crate::render_graph::RenderGraph;
use std::rc::Rc;
use std::time::Instant;
use wgpu::{util::StagingBelt, Buffer, CommandEncoder, CommandEncoderDescriptor, Device, TextureDescriptor};
use wgpu_profiler::{wgpu_profiler, GpuProfiler};
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, Event, KeyboardInput, MouseButton, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::Window;

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

const DISPATCH: usize = 32;

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct Globals {
    pub resolution: [f32; 2],
    pub scroll_offset: [f32; 2],
    pub zoom: f32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct PosNormVertex {
    pub position: [f32; 2],
    pub normal: [f32; 2],
    pub prim_id: i32,
}

impl GPUVertex for PosNormVertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            label: Some("MSAA Framebuffer"),
        },
    )
}

trait BrushRendererTrait {
    fn update(&self, device: &Device, encoder: &mut CommandEncoder, staging_belt: &mut StagingBelt);
    fn render_node(&self, render_graph: &mut RenderGraph, target: GraphNode) -> GraphNode;
}

struct DocumentRenderer {
    vector_renderer: Arc<Mutex<VectorRenderer>>,
    brush_renderer: Arc<Mutex<dyn RenderNode>>,
}

pub struct BrushRenderer {
    target: Arc<Mutex<dyn RenderNode>>,
    vbo: BufferRange,
    ibo: BufferRange,
    material: Arc<Material>,
    brush_manager: Rc<BrushManager>,
    stroke_ranges: Vec<ContinuousStroke>,
    texture: Arc<Mutex<dyn RenderNode>>,
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

    result.push((sub_path.first().index(), sub_path.first().position()));

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
    target: Arc<Mutex<dyn RenderNode>>,
    ibo: BufferRange,
    vbo: BufferRange,
    brush_manager: Rc<BrushManager>,
    material: Arc<Material>,
    points: Vec<(usize, CanvasPoint)>,
    size: f32,
    width_in_pixels: u32,
    view: CanvasView,
    texture: Arc<Mutex<dyn RenderNode>>,
}

impl BrushRendererWithReadback {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        target: Arc<Mutex<dyn RenderNode>>,
        brush_data: &BrushData,
        view: &CanvasView,
        device: &Device,
        _encoder: &mut CommandEncoder,
        _scene_ubo: &BufferRange,
        brush_manager: &Rc<BrushManager>,
        texture: Arc<Mutex<dyn RenderNode>>,
    ) -> BrushRendererWithReadback {
        let mut points = sample_points_along_curve(&brush_data.path, 1.41);
        if points.len() > 200 {
            points.drain(0..points.len() - 200);
        }

        let to_normalized_pos = |v: CanvasPoint| view.screen_to_normalized(view.canvas_to_screen_point(v));

        let size = 256.0;
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

        let vbo = create_buffer_range(device, &vertices, wgpu::BufferUsages::VERTEX, None);

        #[allow(clippy::identity_op)]
        let indices: Vec<u32> = (0..points.len() as u32)
            .flat_map(|x| vec![4 * x + 0, 4 * x + 1, 4 * x + 2, 4 * x + 3, 4 * x + 2, 4 * x + 0])
            .collect();
        let ibo = create_buffer_range(device, &indices, wgpu::BufferUsages::INDEX, None);

        let sampler = Arc::new(device.create_sampler(&wgpu::SamplerDescriptor {
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
        }));

        let width_in_pixels = (2.0 * (CanvasLength::new(size) * view.canvas_to_screen_scale()).get()).round() as u32;

        let material = Arc::new(Material::from_consecutive_entries(
            "clone brush readback",
            BlendState::REPLACE,
            brush_manager.splat_with_readback.bind_group_layout.clone(),
            vec![
                BindingResourceArc::sampler(Some(sampler)),
                BindingResourceArc::render_texture(None),
                BindingResourceArc::render_texture(None),
            ],
        ));

        Self {
            brush_manager: brush_manager.clone(),
            vbo,
            ibo,
            points,
            size,
            width_in_pixels,
            material,
            view: *view,
            target,
            texture,
        }
    }

    pub fn render(&self, _encoder: &mut Encoder, _view: &CanvasView) {
        // if self.points.len() <= 1 {
        //     return;
        // }

        // let temp_to_frame_blitter =
        //     encoder
        //         .blitter
        //         .with_textures(encoder.device, &self.temp_texture_view, encoder.target_texture.view);

        // let bind_group = encoder.device.create_bind_group(&wgpu::BindGroupDescriptor {
        //     layout: &self.brush_manager.splat_with_readback.bind_group_layout,
        //     label: Some("Clone brush Bind Group"),
        //     entries: &[
        //         wgpu::BindGroupEntry {
        //             binding: 0,
        //             resource: wgpu::BindingResource::Sampler(&self.sampler),
        //         },
        //         wgpu::BindGroupEntry {
        //             binding: 1,
        //             resource: wgpu::BindingResource::TextureView(encoder.target_texture.view),
        //         },
        //         wgpu::BindGroupEntry {
        //             binding: 2,
        //             resource: wgpu::BindingResource::TextureView(&self.brush_texture.view),
        //         },
        //     ],
        // });

        // let to_normalized_pos = |v: CanvasPoint| view.screen_to_normalized(view.canvas_to_screen_point(v));

        // for (mut i, (_, p)) in self.points.iter().enumerate() {
        //     // First point is a noop
        //     if i == 0 {
        //         continue;
        //     }
        //     i -= 1;

        //     // First pass
        //     {
        //         let mut pass = encoder.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        //             label: Some("Brush with readback. First pass."),
        //             color_attachments: &[wgpu::RenderPassColorAttachment {
        //                 view: &self.temp_texture_view,
        //                 ops: wgpu::Operations {
        //                     load: wgpu::LoadOp::Clear(wgpu::Color::RED),
        //                     store: true,
        //                 },
        //                 resolve_target: None,
        //             }],
        //             depth_stencil_attachment: None,
        //         });

        //         pass.set_pipeline(&self.brush_manager.splat_with_readback.pipeline);
        //         pass.set_bind_group(0, &bind_group, &[]);
        //         pass.set_index_buffer(self.ibo.slice(..), wgpu::IndexFormat::Uint32);
        //         pass.set_vertex_buffer(0, self.vbo.slice(..));
        //         pass.draw_indexed((i * 6) as u32..((i + 1) * 6) as u32, 0, 0..1);
        //     }

        //     // Second pass, copy back
        //     {
        //         let mn = to_normalized_pos(*p - vector(self.size, self.size));
        //         let mx = to_normalized_pos(*p + vector(self.size, self.size));
        //         let r = rect(mn.x, mn.y, mx.x - mn.x, mx.y - mn.y);
        //         // let uv = vertices[(i*4)..(i+1)*4].iter().map(|x| x.uv_background_target).collect::<ArrayVec<[Point;4]>>();
        //         // let r = Rect::from_points(uv);
        //         temp_to_frame_blitter.blit(encoder.device, encoder.encoder, rect(0.0, 0.0, 1.0, 1.0), r, 1, None);
        //     }
        // }

        // encoder.blitter.blit(
        //     encoder.device,
        //     encoder.encoder,
        //     encoder.target_texture.view,
        //     encoder.multisampled_render_target.as_ref().unwrap().view,
        //     rect(0.0, 0.0, 1.0, 1.0),
        //     rect(0.0, 0.0, 1.0, 1.0),
        //     8,
        //     None,
        // );
    }
}

#[derive(Copy, Clone, PartialEq)]
pub enum BrushType {
    Normal,
    Smudge,
    SmudgeBatch,
    SmudgeSingle,
}

impl RenderNode for BrushRendererWithReadback {
    fn render_node(&self, graph: &mut PersistentGraph) -> GraphNode {
        let mut target = graph.render(&self.target);
        if self.points.len() <= 1 {
            return target;
        }
        let texture = graph.render(&self.texture);

        let to_screen_pos = |v: CanvasPoint| self.view.canvas_to_screen_point(v);

        for (mut i, (_, p)) in self.points.iter().enumerate() {
            // First point is a noop
            if i == 0 {
                continue;
            }
            i -= 1;

            // First pass
            let mut scratch = graph.render_graph.clear(
                size(self.width_in_pixels, self.width_in_pixels),
                wgpu::Color::TRANSPARENT,
            );
            let mut ibo = self.ibo.clone();
            ibo.range.start += ((i * 6) * std::mem::size_of::<u32>()) as u64;
            ibo.range.end = self.ibo.range.start + (((i + 1) * 6) * std::mem::size_of::<u32>()) as u64;
            let material = DynamicMaterial::new(
                self.material.clone(),
                vec![
                    BindGroupEntryArc {
                        binding: 1,
                        resource: BindingResourceArc::graph_node(target.clone()),
                    },
                    BindGroupEntryArc {
                        binding: 2,
                        resource: BindingResourceArc::graph_node(texture.clone()),
                    },
                ],
            );
            scratch = graph.render_graph.mesh(
                scratch,
                self.vbo.clone(),
                ibo,
                self.brush_manager.splat_with_readback.pipeline.clone(),
                material,
            );

            // Second pass: copy back
            let mn = to_screen_pos(*p - vector(self.size, self.size));
            let mx = to_screen_pos(*p + vector(self.size, self.size));
            let r = CanvasRect::new(mn.cast_unit(), size(mx.x - mn.x, mx.y - mn.y));
            target = graph.render_graph.blit(
                scratch,
                target,
                CanvasRect::new(
                    point(0.0, 0.0),
                    size(self.width_in_pixels as f32, self.width_in_pixels as f32),
                ),
                r,
            );
        }

        target
    }

    fn update(&mut self, _device: &Device, _encoder: &mut CommandEncoder, _staging_belt: &mut StagingBelt) {}
}

pub struct BrushRendererWithReadbackSingle {
    target: Arc<Mutex<dyn RenderNode>>,
    ubo: BufferRange,
    brush_manager: Rc<BrushManager>,
    group_width: u32,
    size_in_pixels: u32,
    num_primitives: usize,
    sampler: Arc<wgpu::Sampler>,
    material: Arc<Material>,
    texture: Arc<Mutex<dyn RenderNode>>,
}

fn pixel_path<U>(mut points: impl Iterator<Item = euclid::Point2D<f32, U>>) -> Vec<euclid::Point2D<i32, U>> {
    let mut prev = match points.next() {
        Some(x) => x,
        None => return vec![],
    }
    .round()
    .cast::<i32>();

    let mut result = vec![prev];
    for p in points {
        let next = p.round().cast::<i32>();
        while prev != next {
            let delta = next - prev;
            if delta.x.abs() > delta.y.abs() {
                prev += vector(delta.x.signum(), 0);
            } else {
                prev += vector(0, delta.y.signum());
            }
            result.push(prev);

            // if delta.x.abs() > delta.y.abs() {
            //     prev += vector(delta.x.signum(), 0);
            //     result.push(prev);
            // } else if delta.y.abs() > delta.x.abs() {
            //     prev += vector(0, delta.y.signum());
            //     result.push(prev);
            // } else {
            //     // Diagonal
            //     prev += vector(delta.x.signum(), delta.y.signum());
            //     result.push(prev);
            // }
        }
    }

    result
}

impl BrushRendererWithReadbackSingle {
    const LOCAL_SIZE: usize = 32;

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        target: Arc<Mutex<dyn RenderNode>>,
        brush_data: &BrushData,
        _view: &CanvasView,
        device: &Device,
        _encoder: &mut CommandEncoder,
        _scene_ubo: &BufferRange,
        brush_manager: &Rc<BrushManager>,
        texture: Arc<Mutex<dyn RenderNode>>,
    ) -> BrushRendererWithReadbackSingle {
        let size_in_pixels = 256;
        let mut primitives = vec![];
        let mut rnd = rand::thread_rng();
        for subpath in brush_data.path.iter_sub_paths() {
            let mut points = vec![];
            sample_points_along_sub_path(&subpath, 2.0, &mut points);
            let points = pixel_path(points.into_iter().map(|(_, p)| p));
            // .chunks_exact(2)
            // .map(|a| a[0])
            // .collect::<Vec<_>>();

            let offset = -vector(size_in_pixels as i32 / 2, size_in_pixels as i32 / 2);
            primitives.extend(points.windows(2).enumerate().map(|(i, window)| {
                let clone_pos = window[0] + offset;
                let pos = window[1] + offset;

                ReadbackPrimitive {
                    origin_src: (clone_pos.x, clone_pos.y),
                    origin_dst: (pos.x, pos.y),
                    start: if i == 0 { 1 } else { 0 },
                    random_offset_64: (rnd.gen_range(0..64), rnd.gen_range(0..64)),
                }
            }));
        }

        let ubo = create_buffer_range(device, &primitives, wgpu::BufferUsages::STORAGE, None);

        let sampler = Arc::new(device.create_sampler(&wgpu::SamplerDescriptor {
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
        }));

        let width_per_group = 1; //(size_in_pixels as usize + Self::LOCAL_SIZE - 1) / Self::LOCAL_SIZE;
        let height_per_group = 1; //(size_in_pixels as usize + Self::LOCAL_SIZE - 1) / Self::LOCAL_SIZE;

        let settings_ubo = create_buffer_range(
            device,
            &[ReadbackUniforms {
                width_per_group: width_per_group as i32,
                height_per_group: height_per_group as i32,
                num_primitives: primitives.len() as i32,
                size_in_pixels: size_in_pixels as i32,
            }],
            wgpu::BufferUsages::UNIFORM,
            "Readback settings",
        );

        let material = Arc::new(Material::from_consecutive_entries(
            "clone brush single",
            BlendState::REPLACE,
            brush_manager.splat_with_readback_single_a.bind_group_layout.clone(),
            vec![
                BindingResourceArc::texture(None),
                BindingResourceArc::texture(None),
                BindingResourceArc::texture(Some(brush_manager.blue_noise_tex.clone())),
                BindingResourceArc::sampler(Some(sampler.clone())),
                BindingResourceArc::texture(None),
                BindingResourceArc::buffer(Some(ubo.clone())),
                BindingResourceArc::buffer(Some(settings_ubo)),
            ],
        ));

        Self {
            group_width: (Self::LOCAL_SIZE * width_per_group) as u32,
            brush_manager: brush_manager.clone(),
            ubo,
            size_in_pixels,
            sampler,
            num_primitives: primitives.len(),
            material,
            target,
            texture,
        }
    }
}

impl RenderNode for BrushRendererWithReadbackSingle {
    fn update(&mut self, _device: &Device, _encoder: &mut CommandEncoder, _staging_belt: &mut StagingBelt) {}

    fn render_node(&self, graph: &mut PersistentGraph) -> GraphNode {
        let target = graph.render(&self.target);
        if self.ubo.size() == 0 {
            return target;
        }

        let texture = graph.render(&self.texture);

        let scratch = graph.render_graph.clear_with_format(
            size(self.size_in_pixels * 2, self.size_in_pixels),
            wgpu::Color::RED,
            wgpu::TextureFormat::Rgba32Float,
        );
        let material = DynamicMaterial::new(
            self.material.clone(),
            vec![
                BindGroupEntryArc {
                    binding: 0,
                    resource: BindingResourceArc::graph_node(target.clone()),
                },
                BindGroupEntryArc {
                    binding: 1,
                    resource: BindingResourceArc::graph_node(scratch.clone()),
                },
                BindGroupEntryArc {
                    binding: 4,
                    resource: BindingResourceArc::graph_node(texture.clone()),
                },
            ],
        );

        let dispatch_width = (self.size_in_pixels + self.group_width as u32 - 1) / self.group_width as u32;
        graph.render_graph.custom_compute(
            vec![],
            vec![target, scratch],
            BrushRendererWithReadbackSingleCPassPrimitive {
                pipeline: self.brush_manager.splat_with_readback_single_a.pipeline.clone(),
                primitive_count: self.num_primitives,
                material: material.into(),
                dispatch: (dispatch_width, dispatch_width, 1),
            },
        )
    }
}

struct BrushRendererWithReadbackSingleCPassPrimitive {
    pipeline: Arc<wgpu::ComputePipeline>,
    primitive_count: usize,
    material: crate::render_graph::RenderGraphMaterial,
    dispatch: (u32, u32, u32),
}

impl crate::render_graph::CustomComputePassPrimitive for BrushRendererWithReadbackSingleCPassPrimitive {
    fn compile<'a>(
        &'a self,
        context: &mut crate::render_graph::CompilationContext<'a>,
    ) -> Box<dyn crate::render_graph::CustomComputePass> {
        let resolved = context.resolve_material(&self.material).clone();
        Box::new(BrushRendererWithReadbackSingleCPassCompiled {
            pipeline: self.pipeline.clone(),
            primitive_count: self.primitive_count,
            bind_group: resolved.bind_group(context.device).to_owned(),
            dispatch: self.dispatch,
        })
    }
}

struct BrushRendererWithReadbackSingleCPassCompiled {
    pipeline: Arc<wgpu::ComputePipeline>,
    primitive_count: usize,
    bind_group: Rc<wgpu::BindGroup>,
    dispatch: (u32, u32, u32),
}

impl crate::render_graph::CustomComputePass for BrushRendererWithReadbackSingleCPassCompiled {
    fn execute<'a>(&'a self, device: &Device, gpu_profiler: &mut GpuProfiler, cpass: &mut wgpu::ComputePass<'a>) {
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &self.bind_group, &[]);
        let (dx, dy, dz) = self.dispatch;
        wgpu_profiler!("smudge", gpu_profiler, cpass, device, {
            for i in 0..self.primitive_count * 2 {
                cpass.set_push_constants(0, as_u8_slice(&[i as u32]));
                cpass.dispatch_workgroups(dx, dy, dz);
            }
        });
    }
}

pub struct BrushRendererWithReadbackBatched {
    target: Arc<Mutex<dyn RenderNode>>,
    ubo: BufferRange,
    brush_manager: Rc<BrushManager>,
    size_in_pixels: u32,
    sampler: Arc<wgpu::Sampler>,
    material: Arc<Material>,
    atomic_sbo: BufferRange,
    readback_buffer: BufferRange,
    atomic_output: BufferRange,
    texture: Arc<Mutex<dyn RenderNode>>,
}

#[repr(C, align(16))]
#[derive(Copy, Clone)]
struct ReadbackPrimitive {
    origin_src: (i32, i32),
    origin_dst: (i32, i32),
    random_offset_64: (u32, u32),
    start: u32,
}

#[repr(C, align(16))]
struct ReadbackUniforms {
    width_per_group: i32,
    height_per_group: i32,
    num_primitives: i32,
    size_in_pixels: i32,
}

impl BrushRendererWithReadbackBatched {
    const LOCAL_SIZE: usize = 32;

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        target: Arc<Mutex<dyn RenderNode>>,
        brush_data: &BrushData,
        _view: &CanvasView,
        device: &Device,
        _encoder: &mut CommandEncoder,
        _scene_ubo: &BufferRange,
        brush_manager: &Rc<BrushManager>,
        texture: Arc<Mutex<dyn RenderNode>>,
    ) -> BrushRendererWithReadbackBatched {
        let size_in_pixels = 256;

        let mut primitives = vec![];
        for subpath in brush_data.path.iter_sub_paths() {
            let mut points = vec![];
            sample_points_along_sub_path(&subpath, 2.0, &mut points);
            let points = pixel_path(points.into_iter().map(|(_, p)| p))
                .chunks_exact(2)
                .map(|a| a[0])
                .collect::<Vec<_>>();

            let offset = -vector(size_in_pixels as i32 / 2, size_in_pixels as i32 / 2);
            primitives.extend(points.windows(2).enumerate().map(|(i, window)| {
                let clone_pos = window[0] + offset;
                let pos = window[1] + offset;

                ReadbackPrimitive {
                    origin_src: (clone_pos.x, clone_pos.y),
                    origin_dst: (pos.x, pos.y),
                    start: if i == 0 { 1 } else { 0 },
                    random_offset_64: (0, 0),
                }
            }));
        }

        let ubo = create_buffer_range(device, &primitives, wgpu::BufferUsages::STORAGE, None);

        let sampler = Arc::new(device.create_sampler(&wgpu::SamplerDescriptor {
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
        }));

        let width_per_group = (size_in_pixels as usize + Self::LOCAL_SIZE - 1) / Self::LOCAL_SIZE;
        let height_per_group = (size_in_pixels as usize + Self::LOCAL_SIZE - 1) / Self::LOCAL_SIZE;

        let settings_ubo = create_buffer_range(
            device,
            &[ReadbackUniforms {
                width_per_group: width_per_group as i32,
                height_per_group: height_per_group as i32,
                num_primitives: primitives.len() as i32,
                size_in_pixels: size_in_pixels as i32,
            }],
            wgpu::BufferUsages::UNIFORM,
            "Readback settings",
        );

        let atomic_sbo = create_buffer_range(
            device,
            &[0u32, 0u32],
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            "atomic counter",
        );
        let atomic_output = create_buffer_range(
            device,
            &vec![0; DISPATCH * DISPATCH * Self::LOCAL_SIZE * Self::LOCAL_SIZE],
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            "atomic output",
        );
        let readback_buffer = create_buffer_range(
            device,
            &vec![0; DISPATCH * DISPATCH * Self::LOCAL_SIZE * Self::LOCAL_SIZE],
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            "readback",
        );

        let material = Arc::new(Material::from_consecutive_entries(
            "clone brush batched",
            BlendState::REPLACE,
            brush_manager.splat_with_readback_batched.bind_group_layout.clone(),
            vec![
                BindingResourceArc::texture(None),
                BindingResourceArc::texture(None),
                BindingResourceArc::sampler(Some(sampler.clone())),
                BindingResourceArc::texture(None),
                BindingResourceArc::buffer(Some(ubo.clone())),
                BindingResourceArc::buffer(Some(settings_ubo)),
                BindingResourceArc::buffer(Some(atomic_sbo.clone())),
                BindingResourceArc::buffer(Some(atomic_output.clone())),
            ],
        ));

        Self {
            brush_manager: brush_manager.clone(),
            ubo,
            size_in_pixels,
            sampler,
            material,
            atomic_sbo,
            readback_buffer,
            atomic_output,
            target,
            texture,
        }
    }

    pub fn readback(&self, encoder: &mut CommandEncoder) {
        encoder.copy_buffer_to_buffer(
            &self.atomic_output.buffer,
            0,
            &self.readback_buffer.buffer,
            0,
            4 * (DISPATCH * DISPATCH * Self::LOCAL_SIZE * Self::LOCAL_SIZE) as u64,
        );
    }

    // pub fn check_readback(&self, device: &Device) {
    //     {
    //         let slice = self.readback_buffer.buffer.slice(..);
    //         let f = slice.map_async(wgpu::MapMode::Read);
    //         device.poll(wgpu::Maintain::Wait);
    //         futures::executor::block_on(f).unwrap();
    //         let data = slice.get_mapped_range();
    //         let result: Vec<u32> = data
    //             .chunks_exact(4)
    //             .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
    //             .collect();

    //         dbg!(result.len());
    //         let mut distinct = vec![0; DISPATCH * DISPATCH * Self::LOCAL_SIZE * Self::LOCAL_SIZE];
    //         for r in result {
    //             distinct[r as usize] += 1;
    //         }
    //         if distinct[2] != 0 {
    //             let mut cnt = 0;
    //             for (i, d) in distinct.iter().enumerate() {
    //                 if *d != 1 {
    //                     println!("Diff at {}: {}", i, d);
    //                     cnt += 1;
    //                     if cnt > 10 {
    //                         break;
    //                     }
    //                 }
    //             }
    //         }
    //     }
    //     self.readback_buffer.buffer.unmap();
    // }
}

impl RenderNode for BrushRendererWithReadbackBatched {
    fn update(&mut self, device: &Device, encoder: &mut CommandEncoder, staging_belt: &mut StagingBelt) {
        staging_belt
            .write_buffer(
                encoder,
                &self.atomic_sbo.buffer,
                0,
                NonZeroU64::new(self.atomic_sbo.size()).unwrap(),
                device,
            )
            .fill(0);
    }

    fn render_node(&self, graph: &mut PersistentGraph) -> GraphNode {
        let target = graph.render(&self.target);
        if self.ubo.size() == 0 {
            return target;
        }

        let texture = graph.render(&self.texture);

        let scratch = graph.render_graph.clear_with_format(
            size(self.size_in_pixels * 2, self.size_in_pixels),
            wgpu::Color::RED,
            wgpu::TextureFormat::Rgba32Float,
        );
        let material = DynamicMaterial::new(
            self.material.clone(),
            vec![
                BindGroupEntryArc {
                    binding: 0,
                    resource: BindingResourceArc::graph_node(target.clone()),
                },
                BindGroupEntryArc {
                    binding: 1,
                    resource: BindingResourceArc::graph_node(scratch),
                },
                BindGroupEntryArc {
                    binding: 3,
                    resource: BindingResourceArc::graph_node(texture),
                },
            ],
        );

        graph.render_graph.compute(
            target,
            self.brush_manager.splat_with_readback_batched.pipeline.clone(),
            material,
            (DISPATCH as u32, DISPATCH as u32, 1),
        )
    }
}

struct ContinuousStroke {
    pub vertex_range: Range<u32>,
    pub bbox: CanvasRectI32,
}

impl BrushRenderer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        target: Arc<Mutex<dyn RenderNode>>,
        brush_data: &BrushData,
        view: &CanvasView,
        device: &Device,
        _encoder: &mut CommandEncoder,
        scene_ubo: &BufferRange,
        brush_manager: &Rc<BrushManager>,
        texture: Arc<Mutex<dyn RenderNode>>,
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

            let bbox = CanvasRect::from_points(
                points
                    .iter()
                    .flat_map(|&(_, v)| [v + vector(-size, -size), v + vector(size, size)]),
            )
            .round_out()
            .to_i32();
            let offset = (-bbox.min().to_f32()).to_vector();

            vertices.extend(points.iter().flat_map(|&(vertex_index, pos)| {
                let color = brush_data.colors[vertex_index / 3].into_format().into_raw();
                ArrayVec::from([
                    BrushGpuVertex {
                        position: pos + vector(-size, -size) + offset,
                        uv: point(0.0, 0.0),
                        color,
                    },
                    BrushGpuVertex {
                        position: pos + vector(size, -size) + offset,
                        uv: point(1.0, 0.0),
                        color,
                    },
                    BrushGpuVertex {
                        position: pos + vector(size, size) + offset,
                        uv: point(1.0, 1.0),
                        color,
                    },
                    BrushGpuVertex {
                        position: pos + vector(-size, size) + offset,
                        uv: point(0.0, 1.0),
                        color,
                    },
                ])
            }));

            let start_triangle = ((start_vertex / 4) * 6) as u32;
            let end_triangle = ((vertices.len() / 4) * 6) as u32;
            stroke_ranges.push(ContinuousStroke {
                bbox,
                vertex_range: start_triangle..end_triangle,
            });
        }

        let vbo = create_buffer_range(device, &vertices, wgpu::BufferUsages::VERTEX, None);

        #[allow(clippy::identity_op)]
        let indices: Vec<u32> = (0..(vertices.len() / 4) as u32)
            .flat_map(|x| vec![4 * x + 0, 4 * x + 1, 4 * x + 2, 4 * x + 3, 4 * x + 2, 4 * x + 0])
            .collect();
        let ibo = create_buffer_range(device, &indices, wgpu::BufferUsages::INDEX, None);

        let view_matrix = view.canvas_to_view_matrix();

        let primitive_ubo = create_buffer_range(
            device,
            &[BrushUniforms {
                mvp_matrix: view_matrix * Matrix4::from_translation([0.0, 0.0, 0.1].into()),
            }],
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            "Brush Primitive UBO",
        );

        let sampler = Arc::new(device.create_sampler(&wgpu::SamplerDescriptor {
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
        }));

        let material = Arc::new(Material::from_consecutive_entries(
            "brush",
            BlendState::ALPHA_BLENDING,
            brush_manager.splat.bind_group_layout.clone(),
            vec![
                BindingResourceArc::buffer(Some(scene_ubo.clone())),
                BindingResourceArc::buffer(Some(primitive_ubo)),
                BindingResourceArc::sampler(Some(sampler)),
                BindingResourceArc::texture(None),
            ],
        ));

        BrushRenderer {
            vbo,
            ibo,
            // ubo,
            // index_buffer_length: indices.len(),
            material,
            brush_manager: brush_manager.clone(),
            stroke_ranges,
            target,
            texture,
        }
    }
}

impl RenderNode for BrushRenderer {
    fn render_node(&self, graph: &mut PersistentGraph) -> GraphNode {
        let mut target = graph.render(&self.target);
        if self.stroke_ranges.is_empty() {
            return target;
        }

        let texture = graph.render(&self.texture);
        let mat = DynamicMaterial::new(
            self.material.clone(),
            vec![BindGroupEntryArc {
                binding: 3,
                resource: BindingResourceArc::graph_node(texture),
            }],
        );
        for stroke_path in &self.stroke_ranges {
            if stroke_path.vertex_range.is_empty() {
                continue;
            }

            let bbox = stroke_path.bbox;
            let min_size = (bbox.width().max(bbox.height()) as u32).next_power_of_two().max(32) as i32;
            let scratch_size = size(min_size, min_size).to_u32();
            // let scratch_size = stroke_path.bbox.size.to_u32().to_untyped();
            let scratch = graph.render_graph.clear(scratch_size, wgpu::Color::TRANSPARENT);

            let mut ibo = self.ibo.clone();
            ibo.range.start += stroke_path.vertex_range.start as u64 * std::mem::size_of::<u32>() as u64;
            ibo.range.end =
                self.ibo.range.start + stroke_path.vertex_range.end as u64 * std::mem::size_of::<u32>() as u64;
            assert!(self.ibo.range.contains(&ibo.range.start));
            assert!(self.ibo.range.contains(&(ibo.range.end - 1)));
            let rendered_strokes = graph.render_graph.mesh(
                scratch,
                self.vbo.clone(),
                ibo,
                self.brush_manager.splat.pipeline.clone(),
                mat.clone(),
            );
            target = graph.render_graph.blend(
                rendered_strokes,
                target,
                rect(0.0, 0.0, scratch_size.width as f32, scratch_size.height as f32),
                rect(
                    stroke_path.bbox.min_x(),
                    stroke_path.bbox.min_y(),
                    scratch_size.width as i32,
                    scratch_size.height as i32,
                )
                .to_f32(),
                BlendState::PREMULTIPLIED_ALPHA_BLENDING,
            );
        }
        target
    }

    fn update(&mut self, _device: &Device, _encoder: &mut CommandEncoder, _staging_belt: &mut StagingBelt) {}
}

impl DocumentRenderer {
    #[allow(clippy::too_many_arguments)]
    fn new(
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
        brush_manager: &Rc<BrushManager>,
        brush: BrushType,
    ) -> Self {
        puffin::profile_function!();
        let vector_renderer = VectorRenderer::new(
            target,
            document,
            view,
            device,
            encoder,
            staging_belt,
            bind_group_layout,
            wireframe,
            render_pipeline,
            wireframe_render_pipeline,
        );
        let vector_renderer = Arc::new(Mutex::new(vector_renderer));

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

        let brush_renderer: Arc<Mutex<dyn RenderNode>> = match brush {
            BrushType::Normal => Arc::new(Mutex::new(BrushRenderer::new(
                vector_renderer.clone(),
                &document.brushes,
                view,
                device,
                encoder,
                &scene_ubo,
                brush_manager,
                brush_manager.brushtex.clone(),
            ))),
            BrushType::Smudge => Arc::new(Mutex::new(BrushRendererWithReadback::new(
                vector_renderer.clone(),
                &document.brushes,
                view,
                device,
                encoder,
                &scene_ubo,
                brush_manager,
                brush_manager.brushtex.clone(),
            ))),
            BrushType::SmudgeBatch => Arc::new(Mutex::new(BrushRendererWithReadbackBatched::new(
                vector_renderer.clone(),
                &document.brushes,
                view,
                device,
                encoder,
                &scene_ubo,
                brush_manager,
                brush_manager.brushtex.clone(),
            ))),
            BrushType::SmudgeSingle => Arc::new(Mutex::new(BrushRendererWithReadbackSingle::new(
                vector_renderer.clone(),
                &document.brushes,
                view,
                device,
                encoder,
                &scene_ubo,
                brush_manager,
                brush_manager.brushtex.clone(),
            ))),
        };

        let mut res = DocumentRenderer {
            vector_renderer,
            brush_renderer,
        };
        res.update(device, encoder, staging_belt);
        res
    }
}

impl RenderNode for DocumentRenderer {
    fn update(&mut self, device: &Device, encoder: &mut CommandEncoder, staging_belt: &mut StagingBelt) {
        self.brush_renderer
            .lock()
            .unwrap()
            .update(device, encoder, staging_belt);
    }

    fn render_node(&self, graph: &mut PersistentGraph) -> GraphNode {
        graph.render(&self.brush_renderer)
    }
}

fn dummy_node() -> Arc<Mutex<dyn RenderNode>> {
    Arc::new(Mutex::new(Clear {
        size: Size2D::new(128, 128),
        color: wgpu::Color {
            r: 1.0,
            g: 0.5,
            b: 0.5,
            a: 1.0,
        },
    }))
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
    let sample_count = 1;

    let t0 = Instant::now();

    let t1 = Instant::now();

    let t3 = Instant::now();

    let mut bg_geometry: VertexBuffers<Point, u16> = VertexBuffers::new();
    let mut tessellator = lyon::tessellation::FillTessellator::new();
    tessellator
        .tessellate_rectangle(
            &euclid::Box2D::new(point(-1.0 * 10.0, -1.0 * 10.0), point(0.0, 0.0)),
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
            brush: BrushType::Normal,
        },
        scene_input: SceneInput {
            captured_drag: CapturedDrag::uncaptured(),
        },
        input: InputManager::default(),
    };

    let instance = wgpu::Instance::new(wgpu::Backends::PRIMARY);
    let adapter = task::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None, // TODO
        force_fallback_adapter: false,
    }))
    .unwrap();

    let (device, queue) = task::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: None,
            features: wgpu::Features::POLYGON_MODE_LINE
                | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
                | wgpu::Features::TIMESTAMP_QUERY
                | wgpu::Features::PUSH_CONSTANTS
                | wgpu::Features::SPIRV_SHADER_PASSTHROUGH,
            limits: wgpu::Limits {
                max_push_constant_size: 12,
                ..Default::default()
            },
        },
        None,
    ))
    .expect("Failed to request device");

    let mut staging_belt = wgpu::util::StagingBelt::new(1024);

    let vs_module = Arc::new(crate::shader::load_wgsl_shader(&device, "shaders/geometry.wgsl"));
    let bg_module = Arc::new(crate::shader::load_wgsl_shader(&device, "shaders/background.wgsl"));
    // let vs_module = load_shader(&device, "shaders/geometry.vert.spv");
    // let fs_module = load_shader(&device, "shaders/geometry.frag.spv");
    // let bg_vs_module = load_shader(&device, "shaders/background.vert.spv");
    // let bg_fs_module = load_shader(&device, "shaders/background.frag.spv");

    let bind_group_layout = Arc::new(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Geometry Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    }));

    let pipeline_layout = Arc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        push_constant_ranges: &[],
        bind_group_layouts: &[&bind_group_layout],
    }));

    let depth_stencil_state = Some(wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth32Float,
        depth_write_enabled: true,
        depth_compare: wgpu::CompareFunction::Greater,
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    });

    let render_pipeline = Arc::new(RenderPipelineBase {
        label: "Main Render Pipeline".to_string(),
        layout: pipeline_layout.clone(),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        module: vs_module,
        vertex_buffer_layout: PosNormVertex::desc(),
        vertex_entry: "vs_main".to_string(),
        fragment_entry: "fs_main".to_string(),
        target_count: 1,
    });

    let mut wireframe_render_pipeline = (*render_pipeline).clone();
    wireframe_render_pipeline.label = "Wireframe Render Pipeline".to_string();
    wireframe_render_pipeline.primitive.polygon_mode = wgpu::PolygonMode::Line;
    let wireframe_render_pipeline = Arc::new(wireframe_render_pipeline);

    let bg_pipeline_base = Arc::new(RenderPipelineBase {
        label: "Background pipeline".to_string(),
        layout: pipeline_layout.clone(),
        module: bg_module.clone(),
        vertex_buffer_layout: Point::desc(),
        vertex_entry: "vs_main".to_string(),
        fragment_entry: "fs_main".to_string(),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        target_count: 1,
    });

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
            targets: &[Some(wgpu::ColorTargetState {
                format: crate::config::TEXTURE_FORMAT,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: depth_stencil_state,
        multisample: wgpu::MultisampleState {
            count: sample_count,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
    });

    let pass_info_bind_group_layout = Arc::new(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("pass info"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: wgpu::BufferSize::new(
                    std::mem::size_of::<crate::render_graph::ShaderPassInfo>() as u64
                ),
            },
            count: None,
        }],
    }));

    let brush_manager = Rc::new(BrushManager::load(
        &device,
        &queue,
        sample_count,
        &pass_info_bind_group_layout,
    ));

    let event_loop = EventLoop::new();
    let window = Window::new(&event_loop).unwrap();
    window.set_inner_size(PhysicalSize::new(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT));
    let size = window.inner_size();

    let mut swap_chain_desc = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: crate::config::FRAMEBUFFER_TEXTURE_FORMAT,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Auto,
    };

    let mut multisampled_render_target = None;
    let multisampled_render_target_document;

    let window_surface = unsafe { instance.create_surface(&window) };
    window_surface.configure(&device, &swap_chain_desc);
    // device.create_swap_chain(&window_surface, &swap_chain_desc);

    let mut depth_texture = None;
    let depth_texture_document;

    {
        let document_extent = wgpu::Extent3d {
            width: editor.document.size.unwrap().width,
            height: editor.document.size.unwrap().height,
            depth_or_array_layers: 1,
        };
        depth_texture_document = Some(Arc::new(Texture::new(
            &device,
            wgpu::TextureDescriptor {
                label: Some("Framebuffer depth"),
                size: document_extent,
                mip_level_count: 1,
                sample_count,
                // array_layer_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            },
        )));

        multisampled_render_target_document = if sample_count > 1 {
            Some(Arc::new(create_multisampled_framebuffer(
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
    let mut document_renderer1: Option<Arc<Mutex<DocumentRenderer>>> = None;
    let mut document_renderer2: Option<DocumentRenderer> = None;

    let mut init_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Init encoder"),
    });

    let blitter = Blitter::new(&device, &mut init_encoder);

    let (bg_vbo, _) = create_buffer(&device, &bg_geometry.vertices, wgpu::BufferUsages::VERTEX, "BG VBO");
    let (bg_ibo, _) = create_buffer(&device, &bg_geometry.indices, wgpu::BufferUsages::INDEX, "BG IBO");

    let bg_ubo_data = &[Primitive {
        color: [1.0, 1.0, 1.0, 1.0],
        mvp_matrix: Matrix4::from_translation([0.0, 0.0, 100.0].into()),
        width: 0.0,
    }];
    let bg_ubo = create_buffer_range(&device, bg_ubo_data, wgpu::BufferUsages::UNIFORM, "BG UBO");

    let globals_ubo = create_buffer_range(
        &device,
        &[Globals { ..Default::default() }],
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        "Globals UBO",
    );

    let canvas_globals_ubo = create_buffer_range(
        &device,
        &[Globals { ..Default::default() }],
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        "Canvas Globals UBO",
    );

    let bg_material_base = Arc::new(Material::from_consecutive_entries(
        "background",
        BlendState::REPLACE,
        bind_group_layout.clone(),
        vec![
            BindingResourceArc::buffer(Some(canvas_globals_ubo.clone())),
            BindingResourceArc::buffer(Some(bg_ubo.clone())),
        ],
    ));

    let screen_bg_material_base = Arc::new(Material::from_consecutive_entries(
        "background",
        BlendState::REPLACE,
        bind_group_layout.clone(),
        vec![
            BindingResourceArc::buffer(Some(globals_ubo.clone())),
            BindingResourceArc::buffer(Some(bg_ubo.clone())),
        ],
    ));

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        label: Some("Background Bind Group"),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(canvas_globals_ubo.as_binding()),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(bg_ubo.as_binding()),
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
    // let font = FontArc::try_from_vec(font).unwrap();
    // let glyph_brush = GlyphBrushBuilder::using_font(font).build(&device, crate::config::TEXTURE_FORMAT);

    let document_extent = wgpu::Extent3d {
        width: editor.document.size.unwrap().width,
        height: editor.document.size.unwrap().height,
        depth_or_array_layers: 1,
    };

    let start_time = Instant::now();

    let mut egui_wrapper = EguiWrapper::new(&device, size, window.scale_factor(), 1);
    let mut render_pipeline_cache = RenderPipelineCache::default();
    let mut ephermal_buffer_cache = EphermalBufferCache::default();
    let mut render_texture_cache = RenderTextureCache::default();
    let mut material_cache = MaterialCache::default();

    let mut gpu_profiler = GpuProfiler::new(4, queue.get_timestamp_period(), device.features());
    let mut enable_profiler = false;

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let _guard = runtime.enter();

    let cache_node = Arc::new(Mutex::new(Cache::new(dummy_node())));
    let blit_node = Arc::new(Mutex::new(BlitNode {
        source: cache_node.clone(),
        target: dummy_node(),
        source_rect: CanvasRect::from_size(Size2D::new(0.0, 0.0)),
        target_rect: CanvasRect::from_size(Size2D::new(0.0, 0.0)),
    }));

    event_loop.run(move |event, _, control_flow| {
        {
            let scene = &mut editor.scene;
            {
                puffin::profile_scope!("event handling");
                egui_wrapper.platform.handle_event(&event);

                if update_inputs(event, control_flow, &mut editor.input, scene) {
                    // keep polling inputs.
                    return;
                }
            }

            let new_time = Instant::now();
            let dt = (new_time.duration_since(last_time)).as_secs_f32();
            last_time = new_time;

            // puffin::profile_scope!("event loop");

            {
                puffin::profile_scope!("input");
                egui_wrapper.platform.update_time(start_time.elapsed().as_secs_f64());
                editor.input.block_egui_captured_input(&egui_wrapper.platform.context());
                update_scene_from_input(scene, &mut editor.input, &mut editor.scene_input, dt);

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
                window_surface.configure(&device, &swap_chain_desc);
                depth_texture = Some(Arc::new(Texture::new(
                    &device,
                    TextureDescriptor {
                        label: Some("Framebuffer depth"),
                        size: window_extent,
                        mip_level_count: 1,
                        sample_count,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Depth32Float,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    },
                )));

                multisampled_render_target = if sample_count > 1 {
                    Some(Arc::new(create_multisampled_framebuffer(
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
                loop {
                    match window_surface.get_current_texture() {
                        Ok(x) => break x,
                        Err(e) => println!("{:#?}", e),
                    }
                }
            };

            let frame = RenderTexture::from(SwapchainImageWrapper::from_swapchain_image(
                swapchain_output,
                &swap_chain_desc,
            ));
            let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Frame encoder"),
            });

            let doc_size = editor.document.size.unwrap();

            {
                puffin::profile_scope!("update renderers");

                let dummy_view = CanvasView {
                    zoom: 1.0,
                    scroll: vector(0.0, 0.0),
                    resolution: PhysicalSize::new(doc_size.width, doc_size.height),
                };

                scene.update_uniform_globals(&device, &mut encoder, &mut staging_belt, &globals_ubo.buffer);
                update_buffer_via_transfer(
                    &device,
                    &mut encoder,
                    &mut staging_belt,
                    &[Globals {
                        resolution: [doc_size.width as f32, doc_size.height as f32],
                        zoom: 1.0,
                        scroll_offset: [0.0, 0.0],
                    }],
                    &canvas_globals_ubo.buffer,
                );

                let hash = editor.document.hash() ^ scene.view.hash();
                if hash != last_hash1 || document_renderer1.is_none() {
                    last_hash1 = hash;
                    document_renderer1 = Some(Arc::new(Mutex::new(DocumentRenderer::new(
                        Arc::new(Mutex::new(Clear {
                            size: editor.document.size.unwrap().cast_unit(),
                            color: wgpu::Color {
                                r: 0.5,
                                g: 0.5,
                                b: 0.5,
                                a: 1.0,
                            },
                        })),
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
                        scene.brush,
                    ))));

                    let mut cache = cache_node.lock().unwrap();
                    cache.source = document_renderer1.as_ref().unwrap().clone();
                    cache.dirty();
                }

                let canvas_in_screen_space =
                    scene
                        .view
                        .canvas_to_screen_rect(rect(0.0, 0.0, doc_size.width as f32, doc_size.height as f32));
                let canvas_in_screen_uv_space =
                    canvas_in_screen_space.scale(1.0 / window_extent.width as f32, 1.0 / window_extent.height as f32);

                {
                    let mut blit = blit_node.lock().unwrap();
                    blit.target = Arc::new(Mutex::new(Clear {
                        size: Size2D::new(frame.size().width, frame.size().height),
                        color: wgpu::Color::GREEN,
                    }));
                    blit.source_rect = CanvasRect::from_size(doc_size.cast());
                    blit.target_rect = canvas_in_screen_space.cast_unit();
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
                    editor.ui_document.size = Some(Size2D::new(frame.size().width, frame.size().height));
                    document_renderer2 = Some(DocumentRenderer::new(
                        blit_node.clone(),
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
                        scene.brush,
                    ));
                }

                document_renderer1
                    .as_mut()
                    .unwrap()
                    .lock()
                    .unwrap()
                    .update(&device, &mut encoder, &mut staging_belt);
                document_renderer2
                    .as_mut()
                    .unwrap()
                    .update(&device, &mut encoder, &mut staging_belt);
            }

            ephermal_buffer_cache.reset();

            let mut render_graph = RenderGraph::default();
            let mut graph = PersistentGraph::new(&mut render_graph);

            {
                let framebuffer = document_renderer2.as_ref().unwrap().render_node(&mut graph);
                graph.render_graph.output_texture(framebuffer, frame.clone());
            }

            // Upload all resources for the GPU.
            let screen_descriptor = egui_wgpu_backend::ScreenDescriptor {
                physical_width: window_extent.width,
                physical_height: window_extent.height,
                scale_factor: window.scale_factor() as f32,
            };

            let doc = &mut editor.document;
            let rendering_done = Arc::new(tokio::sync::Notify::new());

            let egui_output = egui_ui(
                &mut egui_wrapper,
                &mut editor.document,
                &device,
                &mut enable_profiler,
                scene,
                document_renderer1.as_ref().unwrap(),
                &mut graph,
                doc_size,
                &rendering_done,
            );

            wgpu_profiler!("rendering", &mut gpu_profiler, &mut encoder, &device, {
                let mut render_graph_compiler = crate::render_graph::RenderGraphCompiler {
                    device: &device,
                    encoder: &mut encoder,
                    blitter: &blitter,
                    render_pipeline_cache: &mut render_pipeline_cache,
                    ephermal_buffer_cache: &mut ephermal_buffer_cache,
                    render_texture_cache: &mut render_texture_cache,
                    material_cache: &mut material_cache,
                    staging_belt: &mut staging_belt,
                    mipmapper: &mipmapper,
                    gpu_profiler: &mut gpu_profiler,
                    pass_info_bind_group_layout: &pass_info_bind_group_layout,
                };

                let passes = render_graph_compiler.compile(&render_graph);
                render_graph_compiler.render(&passes);
            });

            {
                puffin::profile_scope!("staging_belt::finish");
                staging_belt.finish();
            }

            wgpu_profiler!("egui", &mut gpu_profiler, &mut encoder, &device, {
                egui_wrapper.render(
                    egui_output,
                    &device,
                    frame.default_view().view,
                    &queue,
                    &mut encoder,
                    &screen_descriptor,
                );
            });

            let queue_submit_ns = puffin::now_ns();

            rendering_done.notify_one();

            {
                puffin::profile_scope!("resolve gpu profiler queries");
                // Resolves any queries that might be in flight.
                gpu_profiler.resolve_queries(&mut encoder);
            }

            {
                puffin::profile_scope!("queue.submit");
                queue.submit(std::iter::once(encoder.finish()));
            }

            // Signal to the profiler that the frame is finished.
            gpu_profiler.end_frame().unwrap();

            staging_belt.recall();

            // dbg!(t3.elapsed());

            frame_count += 1.0;
            editor.input.tick_frame();
            // println!("Preparing GPU work = {:?}", t1.elapsed());
            {
                puffin::profile_scope!("wait for device");
                device.poll(wgpu::Maintain::Wait);
            }

            crate::gpu_profiler::process_finished_frame(&mut gpu_profiler, queue_submit_ns);

            drop(render_graph);

            // Make sure we drop the profiling scope before we finish the frame.
            // Otherwise the profiling data will show up in the next frame.
            if let RenderTexture::SwapchainImage(image) = frame {
                match Arc::try_unwrap(image.0) {
                    Ok(image) => image.present(),
                    Err(_) => panic!("The swapchain image is still referenced somewhere"),
                }
            }

            fps_limiter.wait(std::time::Duration::from_secs_f32(1.0 / 60.0));
        }
        profiling::finish_frame!();
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
    scene_input: SceneInput,
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
    pub fn build(&self, builder: &mut lyon::path::path::Builder) {
        for path in self.paths.iter() {
            path.build(builder);
        }
    }

    fn hash(&self) -> u64 {
        puffin::profile_function!();
        let mut h = 0u64;
        for path in self.paths.iter() {
            h = h.wrapping_mul(31) ^ path.hash();
        }
        h = h.wrapping_mul(31) ^ self.brushes.hash();
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

pub struct SceneInput {
    pub captured_drag: CapturedDrag,
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
    pub brush: BrushType,
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

fn egui_ui(
    egui_wrapper: &mut EguiWrapper,
    doc: &mut Document,
    device: &Device,
    enable_profiler: &mut bool,
    scene: &mut SceneParams,
    document_renderer1: &Arc<Mutex<DocumentRenderer>>,
    graph: &mut PersistentGraph,
    doc_size: Size2D<u32, CanvasSpace>,
    rendering_done: &Arc<tokio::sync::Notify>,
) -> egui::FullOutput {
    egui_wrapper.frame(|ctx| {
        egui::SidePanel::left("side").show(&ctx, |ui| {
            ui.label("Hello");
            if ui.button("Clear").clicked() {
                doc.brushes.clear();
            }

            if ui.button("Sample path").clicked() {
                doc.brushes.clear();
                doc.brushes.move_to(point(10.0, 10.0), Srgba::new(0.0, 0.0, 0.0, 0.0));
                doc.brushes.line_to(point(100.0, 100.0), Srgba::new(0.0, 0.0, 0.0, 0.0));
                doc.brushes.line_to(point(200.0, 100.0), Srgba::new(0.0, 0.0, 0.0, 0.0));
            }

            ui.checkbox(enable_profiler, "Profiler");

            ui.radio_value(&mut scene.brush, BrushType::Normal, "Normal");
            ui.radio_value(&mut scene.brush, BrushType::Smudge, "Smudge");
            ui.radio_value(&mut scene.brush, BrushType::SmudgeBatch, "SmudgeBatch");
            ui.radio_value(&mut scene.brush, BrushType::SmudgeSingle, "SmudgeSingle");

            if ui.button("Export").clicked() {
                let canvas = graph.render(&(document_renderer1.clone() as Arc<Mutex<dyn RenderNode>>));
                let buffer = create_buffer_range(
                    &device,
                    &vec![0; doc_size.area() as usize],
                    wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    "Readback",
                );
                let mut buffer_node = graph.render_graph.uninitialized_buffer(buffer.size() as usize);
                buffer_node = graph.render_graph.copy_texture_to_buffer(
                    canvas,
                    buffer_node,
                    Extent3d {
                        width: doc_size.width,
                        height: doc_size.height,
                        depth_or_array_layers: 1,
                    },
                );
                graph.render_graph.output_buffer(buffer_node, buffer.clone());

                let rendering_done = rendering_done.clone();
                tokio::spawn(async move {
                    rendering_done.notified().await;
                    let slice = buffer.buffer.slice(..);
                    let buffer = buffer.clone();
                    slice.map_async(wgpu::MapMode::Read, move |r| {
                        {
                            let data = buffer.buffer.slice(..).get_mapped_range();
                            println!("Got output data: {:#?}", &data[0..10]);
                            image::save_buffer("export.png", &data, doc_size.width, doc_size.height, ColorType::Rgba8)
                                .unwrap();
                        }
                        buffer.buffer.unmap();
                    });
                });
            }
        });

        if *enable_profiler && !puffin_egui::profiler_window(&ctx) {
            *enable_profiler = false;
        }
    })
}

fn update_inputs(
    event: Event<()>,
    control_flow: &mut ControlFlow,
    input: &mut InputManager,
    scene: &mut SceneParams,
) -> bool {
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
        _evt => {
            //println!("{:?}", _evt);
        }
    }

    *control_flow = ControlFlow::Poll;
    true
}

pub fn update_scene_from_input(
    scene: &mut SceneParams,
    input: &mut InputManager,
    scene_input: &mut SceneInput,
    delta_time: f32,
) {
    scene_input.captured_drag.try_recapture(input, MouseButton::Right);
    if scene_input.captured_drag.is_captured(input) {
        let cursor_delta = input.mouse_position_delta();
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

    scene.target_zoom *= f32::powf(1.002, -input.scroll_delta.y);

    //println!(" -- zoom: {}, scroll: {:?}", scene.target_zoom, scene.target_scroll);
    let new_zoom = scene.view.zoom + (scene.target_zoom - scene.view.zoom) / 3.0;
    scene.view.zoom_around_point(input.mouse_position, new_zoom);
    scene.stroke_width = scene.stroke_width + (scene.target_stroke_width - scene.stroke_width) / 5.0;
}
