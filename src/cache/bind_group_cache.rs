use std::{collections::HashMap, sync::Arc};

use wgpu::{BindGroup, BindGroupDescriptor, Device};

use crate::wgpu_utils::create_buffer_range;

pub struct BindGroupCache<T> {
    device: &Device,
    cache: HashMap<T, Arc<BindGroup>>,
}

impl<T: Hash + PartialEq> BindGroupCache<T> {
    pub fn len(&mut self) {
        self.cache.len()
    }

    pub fn get(&mut self, value: T) -> &Arc<BindGroup> {
        self.cache.entry(value).or_insert_with(|| {
            let buffer = create_buffer_range(
                device,
                &[value],
                wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
                Some(std::any::type_name::<T>()),
            );
            Arc::new(self.device.create_bind_group(&BindGroupDescriptor {
                label: Some(std::any::type_name::<T>()),
                layout: self.bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_binding(),
                }],
            }))
        })
    }
}
