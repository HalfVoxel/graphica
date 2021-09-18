use std::sync::Arc;

use crate::cache::render_pipeline_cache::RenderPipelineBase;
use crate::shader::load_shader;
use crate::shader::load_wgsl_shader;
use crate::texture::Texture;
use crate::{geometry_utilities::types::*, vertex::GPUVertex};
use image::GenericImageView;
use lyon::math::*;
use wgpu::util::DeviceExt;
use wgpu::{
    ComputePipeline, Device, Extent3d, PushConstantRange, Queue, RenderPipeline, TextureFormat, TextureUsages,
    TextureViewDimension,
};

type GPURGBA = [u8; 4];

#[repr(C)]
#[derive(Copy, Clone)]
pub struct BrushGpuVertex {
    pub position: CanvasPoint,
    pub uv: Point,
    pub color: GPURGBA,
}

impl GPUVertex for BrushGpuVertex {
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
                    format: wgpu::VertexFormat::Unorm8x4,
                    shader_location: 2,
                },
            ],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct CloneBrushGpuVertex {
    pub position: CanvasPoint,
    pub uv_background_source: Point,
    pub uv_background_target: Point,
    pub uv_brush: Point,
    pub color: GPURGBA,
}

impl GPUVertex for CloneBrushGpuVertex {
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
                    format: wgpu::VertexFormat::Float32x2,
                    shader_location: 2,
                },
                wgpu::VertexAttribute {
                    offset: 24,
                    format: wgpu::VertexFormat::Float32x2,
                    shader_location: 3,
                },
                wgpu::VertexAttribute {
                    offset: 32,
                    format: wgpu::VertexFormat::Unorm8x4,
                    shader_location: 4,
                },
            ],
        }
    }
}

pub struct ShaderBundle {
    pub pipeline: Arc<RenderPipelineBase>,
    pub bind_group_layout: Arc<wgpu::BindGroupLayout>,
}

pub struct ShaderBundleCompute {
    pub pipeline: Arc<ComputePipeline>,
    pub bind_group_layout: Arc<wgpu::BindGroupLayout>,
}

pub struct BrushManager {
    pub splat: ShaderBundle,
    pub splat_with_readback: ShaderBundle,
    pub splat_with_readback_batched: ShaderBundleCompute,
    pub splat_with_readback_single_a: ShaderBundleCompute,
    pub splat_with_readback_single_b: ShaderBundleCompute,
    pub blue_noise_tex: Arc<Texture>,
}

impl BrushManager {
    pub fn load(device: &Device, queue: &Queue, _sample_count: u32) -> BrushManager {
        let blue_noise = image::open("blue_noise_56.png").expect("Could not open blue noise image");
        let size = blue_noise.dimensions();
        let blue_noise = blue_noise
            .into_rgba8()
            .chunks_exact(4)
            .map(|c| c[0])
            .collect::<Vec<_>>();
        let blue_noise_tex = Arc::new(crate::texture::Texture::from_data_u8(
            device,
            queue,
            &blue_noise,
            Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            TextureFormat::R8Uint,
            TextureUsages::STORAGE_BINDING | TextureUsages::COPY_DST,
            Some("blue noise"),
        ));

        let bind_group_layout_brush = Arc::new(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind group layout brush"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler {
                        comparison: false,
                        filtering: true,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        }));

        let brush_module = Arc::new(load_wgsl_shader(device, "shaders/brush.wgsl"));
        let clone_brush_module = Arc::new(load_wgsl_shader(device, "shaders/clone_brush.wgsl"));

        let pipeline_layout_brush = Arc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout_brush],
            push_constant_ranges: &[],
        }));

        let render_pipeline_brush = Arc::new(RenderPipelineBase {
            label: "brush".to_string(),
            layout: pipeline_layout_brush,
            module: brush_module,
            vertex_entry: "vs_main".to_string(),
            fragment_entry: "fs_main".to_string(),
            vertex_buffer_layout: BrushGpuVertex::desc(),
            target_count: 1,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                clamp_depth: false,
                conservative: false,
            },
        });

        let bind_group_layout_clone_brush =
            Arc::new(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Bind group layout clone_brush"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler {
                            comparison: false,
                            filtering: true,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            }));

        let pipeline_layout_clone_brush = Arc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout_clone_brush],
            push_constant_ranges: &[],
        }));

        let render_pipeline_clone_brush = Arc::new(RenderPipelineBase {
            label: "clone brush".to_string(),
            layout: pipeline_layout_clone_brush,
            module: clone_brush_module,
            vertex_entry: "vs_main".to_string(),
            fragment_entry: "fs_main".to_string(),
            vertex_buffer_layout: CloneBrushGpuVertex::desc(),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                clamp_depth: false,
                conservative: false,
            },
            target_count: 1,
        });

        let bind_group_layout_clone_brush_batched =
            Arc::new(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            view_dimension: TextureViewDimension::D2,
                            format: crate::config::TEXTURE_FORMAT,
                            access: wgpu::StorageTextureAccess::ReadWrite,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            view_dimension: TextureViewDimension::D2,
                            format: wgpu::TextureFormat::Rgba32Float,
                            access: wgpu::StorageTextureAccess::ReadWrite,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler {
                            filtering: true,
                            comparison: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: None,
            }));

        let pipeline_layout_clone_brush_batched =
            Arc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bind_group_layout_clone_brush_batched],
                push_constant_ranges: &[],
            }));

        let clone_brush_batched_cs_module = Arc::new(load_shader(device, "shaders/clone_brush_batch.comp.spv"));

        let render_pipeline_descriptor_clone_brush_batched = wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout_clone_brush_batched),
            module: &clone_brush_batched_cs_module,
            entry_point: "main",
        };

        let bind_group_layout_clone_brush_single =
            Arc::new(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            view_dimension: TextureViewDimension::D2,
                            format: crate::config::TEXTURE_FORMAT,
                            access: wgpu::StorageTextureAccess::ReadWrite,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            view_dimension: TextureViewDimension::D2,
                            format: wgpu::TextureFormat::Rgba32Float,
                            access: wgpu::StorageTextureAccess::ReadWrite,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            view_dimension: TextureViewDimension::D2,
                            format: wgpu::TextureFormat::R8Uint,
                            access: wgpu::StorageTextureAccess::ReadOnly,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler {
                            filtering: true,
                            comparison: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: None,
            }));

        let pipeline_layout_clone_brush_single =
            Arc::new(device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bind_group_layout_clone_brush_single],
                push_constant_ranges: &[PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..4,
                }],
            }));

        let clone_brush_single_cs_module = Arc::new(load_shader(device, "shaders/clone_brush_single.comp.spv"));

        let render_pipeline_descriptor_clone_brush_single_a = wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout_clone_brush_single),
            module: &clone_brush_single_cs_module,
            entry_point: "main",
        };

        let render_pipeline_descriptor_clone_brush_single_b = wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout_clone_brush_single),
            module: &clone_brush_single_cs_module,
            entry_point: "main",
        };

        BrushManager {
            blue_noise_tex,
            splat: ShaderBundle {
                bind_group_layout: bind_group_layout_brush,
                pipeline: render_pipeline_brush,
            },
            splat_with_readback: ShaderBundle {
                bind_group_layout: bind_group_layout_clone_brush,
                pipeline: render_pipeline_clone_brush,
            },
            splat_with_readback_batched: ShaderBundleCompute {
                bind_group_layout: bind_group_layout_clone_brush_batched,
                pipeline: Arc::new(device.create_compute_pipeline(&render_pipeline_descriptor_clone_brush_batched)),
            },
            splat_with_readback_single_a: ShaderBundleCompute {
                bind_group_layout: bind_group_layout_clone_brush_single.clone(),
                pipeline: Arc::new(device.create_compute_pipeline(&render_pipeline_descriptor_clone_brush_single_a)),
            },
            splat_with_readback_single_b: ShaderBundleCompute {
                bind_group_layout: bind_group_layout_clone_brush_single,
                pipeline: Arc::new(device.create_compute_pipeline(&render_pipeline_descriptor_clone_brush_single_b)),
            },
        }
    }
}
