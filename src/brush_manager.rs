use crate::geometry_utilities::types::*;
use crate::shader::load_shader;
use lyon::math::*;
use wgpu::{Device, RenderPipeline};

type GPURGBA = [u8; 4];

#[repr(C)]
#[derive(Copy, Clone)]
pub struct BrushGpuVertex {
    pub position: CanvasPoint,
    pub uv: Point,
    pub color: GPURGBA,
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

pub struct ShaderBundle {
    pub pipeline: RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

pub struct BrushManager {
    pub splat: ShaderBundle,
    pub splat_with_readback: ShaderBundle,
}

impl BrushManager {
    pub fn load(device: &Device, sample_count: u32) -> BrushManager {
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
                    ty: wgpu::BindingType::SampledTexture {
                        multisampled: true,
                        dimension: wgpu::TextureViewDimension::D2,
                        component_type: wgpu::TextureComponentType::Float,
                    },
                },
            ],
        });

        let brush_vs_module = load_shader(&device, include_bytes!("./../shaders/brush.vert.spv"));
        let brush_fs_module = load_shader(&device, include_bytes!("./../shaders/brush.frag.spv"));
        let clone_brush_vs_module = load_shader(&device, include_bytes!("./../shaders/clone_brush.vert.spv"));
        let clone_brush_fs_module = load_shader(&device, include_bytes!("./../shaders/clone_brush.frag.spv"));

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

        let bind_group_layout_clone_brush = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind group layout clone_brush"),
            bindings: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Sampler { comparison: false },
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::SampledTexture {
                        multisampled: true,
                        dimension: wgpu::TextureViewDimension::D2,
                        component_type: wgpu::TextureComponentType::Float,
                    },
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::SampledTexture {
                        multisampled: true,
                        dimension: wgpu::TextureViewDimension::D2,
                        component_type: wgpu::TextureComponentType::Float,
                    },
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

        BrushManager {
            splat: ShaderBundle {
                bind_group_layout: bind_group_layout_brush,
                pipeline: device.create_render_pipeline(&render_pipeline_descriptor_brush),
            },
            splat_with_readback: ShaderBundle {
                bind_group_layout: bind_group_layout_clone_brush,
                pipeline: device.create_render_pipeline(&render_pipeline_descriptor_clone_brush),
            },
        }
    }
}
