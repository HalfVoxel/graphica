use crate::shader::load_shader;
use crate::{geometry_utilities::types::*, vertex::GPUVertex};
use lyon::math::*;
use wgpu::{ComputePipeline, Device, RenderPipeline, TextureViewDimension};

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
            step_mode: wgpu::InputStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    format: wgpu::VertexFormat::Float2,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    offset: 8,
                    format: wgpu::VertexFormat::Float2,
                    shader_location: 1,
                },
                wgpu::VertexAttribute {
                    offset: 16,
                    format: wgpu::VertexFormat::Uchar4Norm,
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
            step_mode: wgpu::InputStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    format: wgpu::VertexFormat::Float2,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    offset: 8,
                    format: wgpu::VertexFormat::Float2,
                    shader_location: 1,
                },
                wgpu::VertexAttribute {
                    offset: 16,
                    format: wgpu::VertexFormat::Float2,
                    shader_location: 2,
                },
                wgpu::VertexAttribute {
                    offset: 24,
                    format: wgpu::VertexFormat::Float2,
                    shader_location: 3,
                },
                wgpu::VertexAttribute {
                    offset: 32,
                    format: wgpu::VertexFormat::Uchar4Norm,
                    shader_location: 4,
                },
            ],
        }
    }
}

pub struct ShaderBundle {
    pub pipeline: RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

pub struct ShaderBundleCompute {
    pub pipeline: ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

pub struct BrushManager {
    pub splat: ShaderBundle,
    pub splat_with_readback: ShaderBundle,
    pub splat_with_readback_batched: ShaderBundleCompute,
}

impl BrushManager {
    pub fn load(device: &Device, sample_count: u32) -> BrushManager {
        let bind_group_layout_brush = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind group layout brush"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Sampler {
                        comparison: false,
                        filtering: true,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        multisampled: true,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let brush_vs_module = load_shader(&device, "shaders/brush.vert.spv");
        let brush_fs_module = load_shader(&device, "shaders/brush.frag.spv");
        let clone_brush_vs_module = load_shader(&device, "shaders/clone_brush.vert.spv");
        let clone_brush_fs_module = load_shader(&device, "shaders/clone_brush.frag.spv");

        let pipeline_layout_brush = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout_brush],
            push_constant_ranges: &[],
        });

        let render_pipeline_descriptor_brush = wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout_brush),
            vertex: wgpu::VertexState {
                module: &brush_vs_module,
                entry_point: "main",
                buffers: &[BrushGpuVertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &brush_fs_module,
                entry_point: "main",
                targets: &[wgpu::ColorTargetState {
                    format: crate::config::TEXTURE_FORMAT,
                    color_blend: wgpu::BlendState {
                        src_factor: wgpu::BlendFactor::SrcAlpha,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha_blend: wgpu::BlendState {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                    write_mask: wgpu::ColorWrite::ALL,
                }],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: wgpu::CullMode::None,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
        };

        let bind_group_layout_clone_brush = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind group layout clone_brush"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Sampler {
                        comparison: false,
                        filtering: true,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        multisampled: true,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        multisampled: true,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout_clone_brush = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout_clone_brush],
            push_constant_ranges: &[],
        });

        let render_pipeline_descriptor_clone_brush = wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout_clone_brush),
            vertex: wgpu::VertexState {
                module: &clone_brush_vs_module,
                entry_point: "main",
                buffers: &[CloneBrushGpuVertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &clone_brush_fs_module,
                entry_point: "main",
                targets: &[wgpu::ColorTargetState {
                    format: crate::config::TEXTURE_FORMAT,
                    color_blend: wgpu::BlendState {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::Zero,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha_blend: wgpu::BlendState {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::Zero,
                        operation: wgpu::BlendOperation::Add,
                    },
                    write_mask: wgpu::ColorWrite::ALL,
                }],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: wgpu::CullMode::None,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
        };

        let bind_group_layout_clone_brush_batched = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        view_dimension: TextureViewDimension::D2,
                        format: crate::config::TEXTURE_FORMAT,
                        access: wgpu::StorageTextureAccess::ReadWrite,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStage::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        view_dimension: TextureViewDimension::D2,
                        format: crate::config::TEXTURE_FORMAT,
                        access: wgpu::StorageTextureAccess::ReadWrite,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStage::COMPUTE,
                    ty: wgpu::BindingType::Sampler {
                        filtering: true,
                        comparison: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStage::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStage::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStage::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
            label: None,
        });

        let pipeline_layout_clone_brush_batched = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout_clone_brush_batched],
            push_constant_ranges: &[],
        });

        let clone_brush_batched_cs_module = load_shader(&device, "shaders/clone_brush_batch.comp.spv");

        let render_pipeline_descriptor_clone_brush_batched = wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout_clone_brush_batched),
            module: &clone_brush_batched_cs_module,
            entry_point: "main",
        };

        BrushManager {
            splat: ShaderBundle {
                bind_group_layout: bind_group_layout_brush,
                pipeline: device.create_render_pipeline(&render_pipeline_descriptor_brush),
            },
            splat_with_readback: ShaderBundle {
                bind_group_layout: bind_group_layout_clone_brush,
                pipeline: device.create_render_pipeline(&render_pipeline_descriptor_clone_brush),
            },
            splat_with_readback_batched: ShaderBundleCompute {
                bind_group_layout: bind_group_layout_clone_brush_batched,
                pipeline: device.create_compute_pipeline(&render_pipeline_descriptor_clone_brush_batched),
            },
        }
    }
}
