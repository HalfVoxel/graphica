use std::rc::Rc;
use std::sync::Arc;

use crate::cache::material_cache::{BindingResourceArc, Material};
use crate::cache::render_pipeline_cache::RenderPipelineBase;
use crate::shader::load_wgsl_shader;
use crate::wgpu_utils::*;
use crate::{shader::load_shader, vertex::GPUVertex};
use lyon::math::*;
use wgpu::{
    AddressMode, BindGroup, BindGroupLayout, BlendState, Buffer, CommandEncoder, ComputePipeline,
    ComputePipelineDescriptor, DepthStencilState, Device, FilterMode, PipelineLayoutDescriptor, RenderPass,
    RenderPipeline, RenderPipelineDescriptor, Sampler, SamplerDescriptor, TextureView,
};

#[repr(C)]
#[derive(Copy, Clone)]
pub struct BlitGpuVertex {
    pub uv_source: Point,
    pub uv_target: Point,
}

impl GPUVertex for BlitGpuVertex {
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
            ],
        }
    }
}
pub struct BlitterWithTextures<'a, 'b> {
    blitter: &'a Blitter,
    bind_group: Rc<BindGroup>,
    target_texture: &'b TextureView,
}

pub struct Blitter {
    pub render_pipeline_base: Arc<RenderPipelineBase>,
    pub render_pipelines: Vec<Rc<RenderPipeline>>,
    pub bind_group_layout: Arc<BindGroupLayout>,
    pub bind_group_layout_compute: BindGroupLayout,
    pub bind_group_layout_compute_in_place: BindGroupLayout,
    pub material: Material,
    pub sampler: Arc<Sampler>,
    pub ibo: Buffer,
    render_pipeline_blend_over: ComputePipeline,
    render_pipeline_rgb_to_srgb: ComputePipeline,
}

impl Blitter {
    pub fn new(device: &Device, _encoder: &mut CommandEncoder) -> Blitter {
        let blit_module = Arc::new(load_wgsl_shader(device, "shaders/blit.wgsl"));

        let sampler = Arc::new(device.create_sampler(&SamplerDescriptor {
            label: Some("Blitter sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            lod_min_clamp: -100.0,
            lod_max_clamp: 100.0,
            compare: None,
            anisotropy_clamp: None,
            border_color: None,
        }));

        let bind_group_layout = Arc::new(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind group layout blit"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Sampler {
                        filtering: true,
                        comparison: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        }));
        let pipeline_layout = Arc::new(device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        }));

        let render_pipeline_base = Arc::new(RenderPipelineBase {
            label: "blit pipeline".to_string(),
            layout: pipeline_layout.clone(),
            module: blit_module.clone(),
            vertex_buffer_layout: BlitGpuVertex::desc(),
            vertex_entry: "vs_main".to_string(),
            fragment_entry: "fs_main".to_string(),
            primitive: wgpu::PrimitiveState {
                strip_index_format: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                clamp_depth: false,
                conservative: false,
            },
            target_count: 1,
        });

        let render_pipelines = (&[1, 8])
            .iter()
            .map(|&sample_count| {
                let render_pipeline_descriptor = RenderPipelineDescriptor {
                    label: Some("blit pipeline"),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &blit_module,
                        entry_point: "vs_main",
                        buffers: &[BlitGpuVertex::desc()],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &blit_module,
                        entry_point: "fs_main",
                        targets: &[wgpu::ColorTargetState {
                            format: crate::config::TEXTURE_FORMAT,
                            blend: Some(wgpu::BlendState::REPLACE),
                            write_mask: wgpu::ColorWrite::ALL,
                        }],
                    }),
                    primitive: wgpu::PrimitiveState {
                        strip_index_format: None,
                        polygon_mode: wgpu::PolygonMode::Fill,
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        front_face: wgpu::FrontFace::Ccw,
                        cull_mode: None,
                        clamp_depth: false,
                        conservative: false,
                    },
                    depth_stencil: Some(DepthStencilState {
                        format: wgpu::TextureFormat::Depth32Float,
                        depth_write_enabled: false,
                        depth_compare: wgpu::CompareFunction::Always,
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: wgpu::MultisampleState {
                        count: sample_count,
                        mask: !0,
                        alpha_to_coverage_enabled: false,
                    },
                };
                Rc::new(device.create_render_pipeline(&render_pipeline_descriptor))
            })
            .collect::<Vec<_>>();

        let material = Material::from_consecutive_entries(
            device,
            "blit material",
            BlendState::REPLACE,
            bind_group_layout.clone(),
            vec![
                BindingResourceArc::sampler(Some(sampler.clone())),
                BindingResourceArc::texture(None),
            ],
        );

        let blend_over_module = load_shader(device, "shaders/blend_over.comp.spv");

        let bind_group_layout_compute_in_place = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind group layout in place"),
            entries: &[wgpu::BindGroupLayoutEntry {
                count: None,
                binding: 0,
                visibility: wgpu::ShaderStage::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    view_dimension: wgpu::TextureViewDimension::D2,
                    access: wgpu::StorageTextureAccess::ReadWrite,
                    format: crate::config::TEXTURE_FORMAT,
                },
            }],
        });

        let bind_group_layout_compute = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind group layout blit"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    count: None,
                    binding: 0,
                    visibility: wgpu::ShaderStage::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        view_dimension: wgpu::TextureViewDimension::D2,
                        access: wgpu::StorageTextureAccess::ReadWrite,
                        format: crate::config::TEXTURE_FORMAT,
                    },
                },
                wgpu::BindGroupLayoutEntry {
                    count: None,
                    binding: 1,
                    visibility: wgpu::ShaderStage::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        view_dimension: wgpu::TextureViewDimension::D2,
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: crate::config::TEXTURE_FORMAT,
                    },
                },
            ],
        });

        let pipeline_layout_compute = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout_compute],
            push_constant_ranges: &[],
        });

        let render_pipeline_blend_over = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout_compute),
            module: &blend_over_module,
            entry_point: "main",
        });

        let blend_rgb_to_srgb = load_shader(device, "shaders/rgb_to_srgb.comp.spv");

        let pipeline_layout_compute_in_place = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout_compute_in_place],
            push_constant_ranges: &[],
        });

        let render_pipeline_rgb_to_srgb = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout_compute_in_place),
            module: &blend_rgb_to_srgb,
            entry_point: "main",
        });

        let indices = &[0, 1, 2, 3, 2, 0];
        let (ibo, _ibo_size) = create_buffer(device, indices, wgpu::BufferUsage::INDEX, "Blitter IBO");

        Blitter {
            render_pipeline_base,
            render_pipelines,
            bind_group_layout,
            bind_group_layout_compute,
            bind_group_layout_compute_in_place,
            render_pipeline_blend_over,
            render_pipeline_rgb_to_srgb,
            sampler,
            ibo,
            material,
        }
    }

    pub fn with_textures<'a, 'b>(
        &'a self,
        device: &Device,
        source_texture: &wgpu::TextureView,
        target_texture: &'b wgpu::TextureView,
    ) -> BlitterWithTextures<'a, 'b> {
        puffin::profile_function!();
        let bind_group = Rc::new(device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.bind_group_layout,
            label: Some("Blit bind group"),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(source_texture),
                },
            ],
        }));

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
        resolve_target: Option<&wgpu::TextureView>,
    ) {
        self.with_textures(device, source_texture, target_texture).blit(
            device,
            encoder,
            source_uv_rect,
            target_uv_rect,
            sample_count,
            resolve_target,
        );
    }

    pub fn blend(
        &self,
        device: &Device,
        encoder: &mut CommandEncoder,
        source_texture: &wgpu::TextureView,
        target_texture: &wgpu::TextureView,
        size: (u32, u32),
    ) {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.bind_group_layout_compute,
            label: Some("Blit bind group"),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source_texture),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(target_texture),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("blend") });
        const LOCAL_SIZE: u32 = 8;
        pass.set_pipeline(&self.render_pipeline_blend_over);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch(
            (size.0 + LOCAL_SIZE - 1) / LOCAL_SIZE,
            (size.1 + LOCAL_SIZE - 1) / LOCAL_SIZE,
            1,
        );
    }

    pub fn rgb_to_srgb(
        &self,
        device: &Device,
        encoder: &mut CommandEncoder,
        texture: &wgpu::TextureView,
        size: (u32, u32),
    ) {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.bind_group_layout_compute_in_place,
            label: Some("Blit bind group"),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(texture),
            }],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rgb to srgb"),
        });
        const LOCAL_SIZE: u32 = 8;
        pass.set_pipeline(&self.render_pipeline_rgb_to_srgb);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch(
            (size.0 + LOCAL_SIZE - 1) / LOCAL_SIZE,
            (size.1 + LOCAL_SIZE - 1) / LOCAL_SIZE,
            1,
        );
    }

    pub fn blit_pipeline(&self, sample_count: u32) -> Rc<RenderPipeline> {
        match sample_count {
            1 => self.render_pipelines[0].clone(),
            8 => self.render_pipelines[1].clone(),
            _ => panic!("Unsupported blit sample count. Only 1 and 8 supported."),
        }
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
        resolve_target: Option<&wgpu::TextureView>,
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

        let (vbo, _) = create_buffer(device, vertices, wgpu::BufferUsage::VERTEX, None);
        let pipeline = self.blitter.blit_pipeline(sample_count);

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("blit"),
            color_attachments: &[wgpu::RenderPassColorAttachment {
                view: self.target_texture,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: true,
                },
                resolve_target,
            }],
            depth_stencil_attachment: None,
        });

        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_index_buffer(self.blitter.ibo.slice(..), wgpu::IndexFormat::Uint32);
        pass.set_vertex_buffer(0, vbo.slice(..));
        pass.draw_indexed(0..6, 0, 0..1);
    }

    pub fn blit_regions(
        &self,
        device: &Device,
        source_uv_rect: Rect,
        target_uv_rect: Rect,
        sample_count: u32,
    ) -> BlitOp<'a> {
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

        let (vbo, _) = create_buffer(device, vertices, wgpu::BufferUsage::VERTEX, None);

        let pipeline: &'a RenderPipeline = match sample_count {
            1 => &self.blitter.render_pipelines[0],
            8 => &self.blitter.render_pipelines[1],
            _ => panic!("Unsupported blit sample count. Only 1 and 8 supported."),
        };

        BlitOp {
            blitter: self.blitter,
            bind_group: self.bind_group.clone(),
            pipeline,
            vbo,
        }
    }
}

pub struct BlitOp<'a> {
    blitter: &'a Blitter,
    pipeline: &'a RenderPipeline,
    bind_group: Rc<BindGroup>,
    vbo: Buffer,
}

impl<'a> BlitOp<'a> {
    pub fn render(&'a self, pass: &mut RenderPass<'a>) {
        puffin::profile_function!();
        pass.set_pipeline(self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_index_buffer(self.blitter.ibo.slice(..), wgpu::IndexFormat::Uint32);
        pass.set_vertex_buffer(0, self.vbo.slice(..));
        pass.draw_indexed(0..6, 0, 0..1);
    }
}
