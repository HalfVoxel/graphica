use crate::shader::load_shader;
use crate::wgpu_utils::*;
use lyon::math::*;
use wgpu::{
    AddressMode, BindGroup, BindGroupLayout, BlendFactor, Buffer, CommandEncoder, Device, FilterMode,
    PipelineLayoutDescriptor, RenderPipeline, RenderPipelineDescriptor, Sampler, SamplerDescriptor, TextureView,
};

#[repr(C)]
#[derive(Copy, Clone)]
struct BlitGpuVertex {
    uv_source: Point,
    uv_target: Point,
}

pub struct BlitterWithTextures<'a, 'b> {
    blitter: &'a Blitter,
    bind_group: BindGroup,
    target_texture: &'b TextureView,
}

pub struct Blitter {
    render_pipelines: Vec<RenderPipeline>,
    bind_group_layout: BindGroupLayout,
    sampler: Sampler,
    ibo: Buffer,
}

impl Blitter {
    pub fn new(device: &Device, encoder: &mut CommandEncoder) -> Blitter {
        let blit_vs = load_shader(&device, include_bytes!("./../shaders/blit.vert.spv"));
        let blit_fs = load_shader(&device, include_bytes!("./../shaders/blit.frag.spv"));

        let sampler = device.create_sampler(&SamplerDescriptor {
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Nearest,
            mipmap_filter: FilterMode::Nearest,
            lod_min_clamp: -100.0,
            lod_max_clamp: 100.0,
            compare: wgpu::CompareFunction::Always,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind group layout blit"),
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
                        multisampled: false,
                        dimension: wgpu::TextureViewDimension::D2,
                        component_type: wgpu::TextureComponentType::Float,
                    },
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            bind_group_layouts: &[&bind_group_layout],
        });

        let render_pipelines = (&[1, 8])
            .iter()
            .map(|&sample_count| {
                let render_pipeline_descriptor = RenderPipelineDescriptor {
                    layout: &pipeline_layout,
                    vertex_stage: wgpu::ProgrammableStageDescriptor {
                        module: &blit_vs,
                        entry_point: "main",
                    },
                    fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                        module: &blit_fs,
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
                            src_factor: BlendFactor::One,
                            dst_factor: BlendFactor::Zero,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha_blend: wgpu::BlendDescriptor {
                            src_factor: BlendFactor::One,
                            dst_factor: BlendFactor::Zero,
                            operation: wgpu::BlendOperation::Add,
                        },
                        write_mask: wgpu::ColorWrite::ALL,
                    }],
                    depth_stencil_state: None,
                    vertex_state: wgpu::VertexStateDescriptor {
                        index_format: wgpu::IndexFormat::Uint32,
                        vertex_buffers: &[wgpu::VertexBufferDescriptor {
                            stride: std::mem::size_of::<BlitGpuVertex>() as u64,
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
                            ],
                        }],
                    },
                    sample_count: sample_count,
                    sample_mask: !0,
                    alpha_to_coverage_enabled: false,
                };
                let render_pipeline = device.create_render_pipeline(&render_pipeline_descriptor);
                render_pipeline
            })
            .collect::<Vec<RenderPipeline>>();

        let indices = &[0, 1, 2, 3, 2, 0];
        let (ibo, ibo_size) =
            create_buffer_via_transfer(device, encoder, indices, wgpu::BufferUsage::INDEX, "Blitter IBO");

        Blitter {
            render_pipelines,
            bind_group_layout,
            sampler,
            ibo,
        }
    }

    pub fn with_textures<'a, 'b>(
        &'a self,
        device: &Device,
        source_texture: &wgpu::TextureView,
        target_texture: &'b wgpu::TextureView,
    ) -> BlitterWithTextures<'a, 'b> {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.bind_group_layout,
            label: Some("Blit bind group"),
            bindings: &[
                wgpu::Binding {
                    binding: 0,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::Binding {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(source_texture),
                },
            ],
        });

        BlitterWithTextures {
            blitter: self,
            bind_group,
            target_texture,
        }
    }

    pub fn blit(
        &self,
        device: &Device,
        encoder: &mut CommandEncoder,
        source_texture: &wgpu::TextureView,
        target_texture: &wgpu::TextureView,
        source_uv_rect: Rect,
        target_uv_rect: Rect,
        sample_count: u32,
    ) {
        self.with_textures(device, source_texture, target_texture).blit(
            device,
            encoder,
            source_uv_rect,
            target_uv_rect,
            sample_count,
        );
    }
}

impl<'a, 'b> BlitterWithTextures<'a, 'b> {
    pub fn blit(
        &self,
        device: &Device,
        encoder: &mut CommandEncoder,
        source_uv_rect: Rect,
        target_uv_rect: Rect,
        sample_count: u32,
    ) {
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

        let (vbo, _) = create_buffer_with_data(&device, vertices, wgpu::BufferUsage::VERTEX);

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                attachment: self.target_texture,
                load_op: wgpu::LoadOp::Load,
                store_op: wgpu::StoreOp::Store,
                clear_color: wgpu::Color::BLACK,
                resolve_target: None,
            }],
            depth_stencil_attachment: None,
        });

        let pipeline = match sample_count {
            1 => &self.blitter.render_pipelines[0],
            8 => &self.blitter.render_pipelines[1],
            _ => panic!("Unsupported blit sample count. Only 1 and 8 supported."),
        };
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_index_buffer(&self.blitter.ibo, 0, 0);
        pass.set_vertex_buffer(0, &vbo, 0, 0);
        pass.draw_indexed(0..6, 0, 0..1);
    }
}
