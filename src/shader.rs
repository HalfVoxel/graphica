use wgpu::{Device, ShaderModule};

pub fn load_shader(device: &Device, shader_bytes: &[u8]) -> ShaderModule {
    let spv = wgpu::read_spirv(std::io::Cursor::new(&shader_bytes)).unwrap();
    device.create_shader_module(&spv)
}
