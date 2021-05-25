use lyon::math::Point;

pub trait GPUVertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a>;
}

impl GPUVertex for Point {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as u64,
            step_mode: wgpu::InputStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                format: wgpu::VertexFormat::Float32x2,
                shader_location: 0,
            }],
        }
    }
}
