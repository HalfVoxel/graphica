use std::{collections::HashMap, num::NonZeroU64, ops::Range, sync::Arc};

use wgpu::{util::StagingBelt, Buffer, BufferAddress, BufferDescriptor, BufferUsage, CommandEncoder, Device};

use crate::wgpu_utils::{as_u8_slice, create_buffer};

#[derive(Default)]
pub struct EphermalBufferCache {
    chunks: HashMap<BufferUsage, Vec<Chunk>>,
}

#[derive(Debug, Clone)]
pub struct BufferRange {
    pub buffer: Arc<Buffer>,
    pub range: Range<BufferAddress>,
}

impl BufferRange {
    pub fn as_slice(&self) -> wgpu::BufferSlice {
        self.buffer.slice(self.range.clone())
    }

    pub fn as_binding(&self) -> wgpu::BufferBinding {
        wgpu::BufferBinding {
            buffer: &self.buffer,
            offset: self.range.start,
            size: Some(NonZeroU64::new(self.range.end - self.range.start).expect("tried to bind zero size buffer range")),
        }
    }

    pub fn size(&self) -> u64 {
        // Range<u64> has no len() implementation
        self.range.end - self.range.start
    }
}

struct Chunk {
    buffer: Arc<Buffer>,
    used: u64,
    size: u64,
}

const ENABLE_CACHE: bool = true;

impl EphermalBufferCache {
    pub fn get<T>(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        staging_belt: &mut StagingBelt,
        mut usage: BufferUsage,
        contents: &[T],
    ) -> BufferRange {
        let bytes = as_u8_slice(contents);
        let content_size = bytes.len() as u64;

        if !ENABLE_CACHE {
            let (vbo, _) = create_buffer(device, contents, usage, None);
            return BufferRange {
                buffer: Arc::new(vbo),
                range: 0..content_size,
            };
        }

        // We need to be able to copy to the buffer in order to be able to use the staging belt
        usage |= BufferUsage::COPY_DST;

        let chunks = self.chunks.entry(usage).or_default();
        let chunk_idx = if let Some(idx) = chunks.iter().position(|chunk| chunk.used + content_size <= chunk.size) {
            idx
        } else {
            let new_size = (content_size)
                .next_power_of_two()
                .max(chunks.last().map(|x| x.size).unwrap_or(0));
            println!("Creating a new buffer with size={}", new_size);
            chunks.push(Chunk {
                buffer: Arc::new(device.create_buffer(&BufferDescriptor {
                    label: Some("ephermal buffer"),
                    size: new_size,
                    usage,
                    mapped_at_creation: false,
                })),
                used: 0,
                size: new_size,
            });

            chunks.len() - 1
        };

        let chunk = &mut chunks[chunk_idx];
        if let Some(size) = NonZeroU64::new(content_size) {
            staging_belt
                .write_buffer(encoder, &chunk.buffer, chunk.used, size, device)
                .copy_from_slice(bytes);
        }

        let result = BufferRange {
            buffer: chunk.buffer.clone(),
            range: chunk.used..chunk.used + content_size,
        };

        chunk.used += content_size;
        // Round up to the next multiple of 8
        // TODO: Investigate alignment requirements
        let remainder = chunk.used % 8;
        if remainder != 0 {
            chunk.used += 8 - remainder;
        }

        result
    }

    pub fn reset(&mut self) {
        for chunks in self.chunks.values_mut() {
            for chunk in chunks {
                chunk.used = 0;
            }
        }
    }
}
