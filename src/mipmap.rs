use crate::main::Texture;
use crate::shader::load_shader;
use wgpu::{
    BindGroupLayout, CommandEncoder, ComputePipeline, Device, Extent3d, TextureComponentType, TextureFormat,
    TextureViewDescriptor, TextureViewDimension,
};

pub struct Mipmapper {
    pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
}

pub fn max_mipmaps(texture_size: Extent3d) -> u32 {
    (texture_size.width as f32)
        .log2()
        .max((texture_size.width as f32).log2())
        .floor() as u32
        + 1
}

impl Mipmapper {
    pub fn new(device: &Device) -> Mipmapper {
        let shader = load_shader(device, include_bytes!("../shaders/downsample_2x2_box.comp.spv"));

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            bindings: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        dimension: TextureViewDimension::D2,
                        component_type: TextureComponentType::Float,
                        format: TextureFormat::Bgra8Unorm,
                        readonly: true,
                    },
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStage::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        dimension: TextureViewDimension::D2,
                        component_type: TextureComponentType::Float,
                        format: TextureFormat::Bgra8Unorm,
                        readonly: false,
                    },
                },
            ],
            label: None,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&bind_group_layout],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            layout: &pipeline_layout,
            compute_stage: wgpu::ProgrammableStageDescriptor {
                module: &shader,
                entry_point: "main",
            },
        });

        Mipmapper {
            pipeline,
            bind_group_layout,
        }
    }

    pub fn generate_mipmaps(&self, device: &Device, encoder: &mut CommandEncoder, texture: &Texture) {
        // assert_eq!(texture.descriptor.dimension, wgpu::TextureViewDimension::D2);
        assert_eq!(texture.descriptor.array_layer_count, 1);
        assert!(texture.descriptor.mip_level_count > 1);
        // assert!((texture.descriptor.size.width & (texture.descriptor.size.width - 1)) == 0, "Texture width must be a power of two. Found {}", texture.descriptor.size.width);
        // assert!((texture.descriptor.size.height & (texture.descriptor.size.height - 1)) == 0, "Texture height must be a power of two. Found {}", texture.descriptor.size.height);

        let mut prev_level = texture.buffer.create_view(&TextureViewDescriptor {
            format: texture.descriptor.format,
            dimension: wgpu::TextureViewDimension::D2,
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            array_layer_count: 1,
        });
        let mut width = texture.descriptor.size.width;
        let mut height = texture.descriptor.size.height;

        let mut bind_groups = vec![];
        let mut cpass = encoder.begin_compute_pass();
        cpass.set_pipeline(&self.pipeline);

        for mip_level in 1..texture.descriptor.mip_level_count {
            let current_level = texture.buffer.create_view(&TextureViewDescriptor {
                format: texture.descriptor.format,
                dimension: wgpu::TextureViewDimension::D2,
                aspect: wgpu::TextureAspect::All,
                base_mip_level: mip_level,
                level_count: 1,
                base_array_layer: 0,
                array_layer_count: 1,
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.bind_group_layout,
                bindings: &[
                    wgpu::Binding {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&prev_level),
                    },
                    wgpu::Binding {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&current_level),
                    },
                ],
                label: None,
            });
            bind_groups.push(bind_group);
            prev_level = current_level;
        }

        for bind_group in &bind_groups {
            width = (width / 2).max(1);
            height = (height / 2).max(1);
            let local_size: u32 = 8;
            {
                cpass.set_bind_group(0, bind_group, &[]);
                cpass.dispatch(
                    (width + local_size - 1) / local_size,
                    (height + local_size - 1) / local_size,
                    1,
                );
            }
        }
    }
}
