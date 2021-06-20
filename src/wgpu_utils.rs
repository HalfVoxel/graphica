use wgpu::{
    util::{DeviceExt, StagingBelt},
    Buffer, BufferSize, CommandEncoder, Device,
};

pub fn as_u8_slice<T>(v: &[T]) -> &[u8] {
    assert!(
        std::mem::size_of::<T>() != 0,
        "You should not use zero sized type for buffers"
    );
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
    puffin::profile_function!();
    let data = as_u8_slice(data);
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: label.into(),
        contents: &data,
        usage,
    });

    (buffer, data.len() as u64)
}

pub fn update_buffer_via_transfer<T>(
    device: &Device,
    encoder: &mut CommandEncoder,
    staging_belt: &mut StagingBelt,
    v: &[T],
    target_buffer: &Buffer,
) {
    puffin::profile_function!();
    if !v.is_empty() {
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
    }
}
