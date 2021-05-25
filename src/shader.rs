use std::io::Read;
use std::path::PathBuf;
use wgpu::{Device, ShaderModule};

pub fn load_wgsl_shader(device: &Device, path: &str) -> ShaderModule {
    let mut file = std::fs::File::open(PathBuf::from(path)).unwrap();
    let mut buf = vec![];
    file.read_to_end(&mut buf).unwrap();
    let text = String::from_utf8(buf).unwrap();
    device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: Some(path),
        source: wgpu::ShaderSource::Wgsl(text.into()),
        flags: wgpu::ShaderFlags::VALIDATION,
    })
}

pub fn load_shader(device: &Device, path: &str) -> ShaderModule {
    let mut file = std::fs::File::open(PathBuf::from(path)).unwrap();
    let mut buf = vec![];
    file.read_to_end(&mut buf).unwrap();
    load_shader_bytes(device, &buf, Some(path))
}

pub fn load_shader_bytes(device: &Device, shader_bytes: &[u8], label: Option<&str>) -> ShaderModule {
    println!("Loading shader {:?}", label);
    let source = wgpu::util::make_spirv(shader_bytes);
    device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label,
        source,
        flags: wgpu::ShaderFlags::empty(),
    })
}
