use wgpu::{
    util::{DeviceExt, StagingBelt},
    Buffer, BufferSize, CommandEncoder, Device,
};

fn as_u8_slice<T>(v: &[T]) -> &[u8] {
    let (head, body, tail) = unsafe { v.align_to::<u8>() };
    assert!(head.is_empty());
    assert!(tail.is_empty());
    body
}

pub fn create_buffer<'a, T>(
    device: &Device,
    data: &[T],
    usage: wgpu::BufferUsage,
    label: impl Into<Option<&'a str>>,
) -> (Buffer, u64) {
    let mut data = as_u8_slice(data);
    let orig_length = data.len() as u64;
    if orig_length == 0 {
        data = &[0, 0, 0, 0];
    }
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: label.into(),
        contents: &data,
        usage,
    });

    (buffer, orig_length)
}

pub fn create_buffer_with_data<'a, T>(device: &Device, v: &[T], usage: wgpu::BufferUsage) -> (Buffer, u64) {
    let mut data = as_u8_slice(v);
    let orig_length = data.len() as u64;
    if orig_length == 0 {
        data = &[0, 0, 0, 0];
    }
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: &data,
        usage,
    });

    (buffer, orig_length)
}

pub fn update_buffer_via_transfer<T>(
    device: &Device,
    encoder: &mut CommandEncoder,
    staging_belt: &mut StagingBelt,
    v: &[T],
    target_buffer: &Buffer,
) {
    let data = as_u8_slice(v);
    staging_belt
        .write_buffer(
            encoder,
            target_buffer,
            0,
            BufferSize::new(data.len() as u64).expect("buffer was empty"),
            &device,
        )
        .copy_from_slice(data);
    // let transfer_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
    //     label: None,
    //     contents: &data,
    //     usage: wgpu::BufferUsage::COPY_SRC,
    // });
    // encoder.copy_buffer_to_buffer(&transfer_buffer, 0, &target_buffer, 0, data.len() as u64);
}
