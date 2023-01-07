use crate::encoder::Encoder;
use crate::shader::load_shader;
use wgpu::{BindGroupLayout, ComputePipeline, Device, TextureViewDimension};

pub struct BlurCompute {
    pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
}

impl BlurCompute {
    pub fn new(device: &Device) -> BlurCompute {
        let cs_module = load_shader(device, "shaders/blur.comp.spv");

        // let staging_buffer = device.create_buffer_with_data(
        //     bytemuck::cast_slice(&numbers),
        //     wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        // );

        // let storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        //     size,
        //     usage: wgpu::BufferUsages::STORAGE
        //         | wgpu::BufferUsages::COPY_DST
        //         | wgpu::BufferUsages::COPY_SRC,
        //     label: None,
        // });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    view_dimension: TextureViewDimension::D2,
                    format: crate::config::TEXTURE_FORMAT,
                    access: wgpu::StorageTextureAccess::ReadWrite,
                },
                count: None,
            }],
            label: None,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &cs_module,
            entry_point: "main",
        });

        BlurCompute {
            pipeline,
            bind_group_layout,
        }
    }

    #[allow(dead_code)]
    pub fn render(&self, encoder: &mut Encoder) {
        let bind_group = encoder.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(encoder.target_texture.view),
            }],
            label: None,
        });

        // encoder.encoder.copy_buffer_to_buffer(&staging_buffer, 0, &storage_buffer, 0, size);
        let local_size: u32 = 32;
        {
            let mut cpass = encoder
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("blur") });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(
                (encoder.resolution.width + local_size - 1) / local_size,
                (encoder.resolution.height + local_size - 1) / local_size,
                1,
            );
        }
        // encoder.encoder.copy_buffer_to_buffer(&storage_buffer, 0, &staging_buffer, 0, size);
    }
}
