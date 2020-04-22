use wgpu::{Buffer, CommandEncoder, Device};

fn as_u8_slice<T>(v: &[T]) -> &[u8] {
    let (head, body, tail) = unsafe { v.align_to::<u8>() };
    assert!(head.is_empty());
    assert!(tail.is_empty());
    body
}

pub fn create_buffer_via_transfer<'a, T>(
    device: &Device,
    encoder: &mut CommandEncoder,
    v: &[T],
    usage: wgpu::BufferUsage,
    label: impl Into<Option<&'a str>>,
) -> (Buffer, u64) {
    let mut data = as_u8_slice(v);
    let orig_length = data.len() as u64;
    if orig_length == 0 {
        data = &[0, 0, 0, 0];
    }
    let transfer_buffer = device.create_buffer_with_data(data, wgpu::BufferUsage::COPY_SRC);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        size: data.len() as u64,
        usage: usage | wgpu::BufferUsage::COPY_DST,
        label: label.into(),
    });
    encoder.copy_buffer_to_buffer(&transfer_buffer, 0, &buffer, 0, data.len() as u64);

    (buffer, orig_length)
}

pub fn create_buffer_with_data<'a, T>(device: &Device, v: &[T], usage: wgpu::BufferUsage) -> (Buffer, u64) {
    let mut data = as_u8_slice(v);
    let orig_length = data.len() as u64;
    if orig_length == 0 {
        data = &[0, 0, 0, 0];
    }
    let buffer = device.create_buffer_with_data(data, usage);

    (buffer, orig_length)
}

pub fn update_buffer_via_transfer<T>(device: &Device, encoder: &mut CommandEncoder, v: &[T], target_buffer: &Buffer) {
    let data = as_u8_slice(v);
    let transfer_buffer = device.create_buffer_with_data(data, wgpu::BufferUsage::COPY_SRC);
    encoder.copy_buffer_to_buffer(&transfer_buffer, 0, &target_buffer, 0, data.len() as u64);
}
