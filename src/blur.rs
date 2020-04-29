use crate::encoder::Encoder;
use crate::shader::load_shader;
use wgpu::{BindGroupLayout, ComputePipeline, Device, TextureComponentType, TextureFormat, TextureViewDimension};

pub struct BlurCompute {
    pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
}

impl BlurCompute {
    pub fn new(device: &Device) -> BlurCompute {
        let cs_module = load_shader(device, include_bytes!("../shaders/blur.comp.spv"));

        // let staging_buffer = device.create_buffer_with_data(
        //     bytemuck::cast_slice(&numbers),
        //     wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::COPY_SRC,
        // );

        // let storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        //     size,
        //     usage: wgpu::BufferUsage::STORAGE
        //         | wgpu::BufferUsage::COPY_DST
        //         | wgpu::BufferUsage::COPY_SRC,
        //     label: None,
        // });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            bindings: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStage::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    dimension: TextureViewDimension::D2,
                    component_type: TextureComponentType::Float,
                    format: crate::config::TEXTURE_FORMAT,
                    readonly: false,
                },
            }],
            label: None,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&bind_group_layout],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            layout: &pipeline_layout,
            compute_stage: wgpu::ProgrammableStageDescriptor {
                module: &cs_module,
                entry_point: "main",
            },
        });

        BlurCompute {
            pipeline,
            bind_group_layout,
        }
    }

    pub fn render(&self, encoder: &mut Encoder) {
        let bind_group = encoder.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.bind_group_layout,
            bindings: &[wgpu::Binding {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&encoder.target_texture),
            }],
            label: None,
        });

        // encoder.encoder.copy_buffer_to_buffer(&staging_buffer, 0, &storage_buffer, 0, size);
        let local_size: u32 = 32;
        {
            let mut cpass = encoder.encoder.begin_compute_pass();
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch(
                (encoder.resolution.width + local_size - 1) / local_size,
                (encoder.resolution.height + local_size - 1) / local_size,
                1,
            );
        }
        // encoder.encoder.copy_buffer_to_buffer(&storage_buffer, 0, &staging_buffer, 0, size);
    }
}
