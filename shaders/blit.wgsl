struct VertexOutput {
    @location(1) uv: vec2<f32>,
    @builtin(position) position: vec4<f32>,
};

@group(0) @binding(0)
var samp: sampler;

@group(0) @binding(1)
var source: texture_2d<f32>;

@vertex
fn vs_main(
    @location(0) source_uv: vec2<f32>,
    @location(1) target_uv: vec2<f32>,
    ) -> VertexOutput {
    let z = 1.0;
    var transformed_pos: vec2<f32> = target_uv * 2.0 - 1.0;
    transformed_pos.y = -transformed_pos.y;
    var out: VertexOutput;
    out.position = vec4<f32>(transformed_pos, z / 1000.0, 1.0);
    out.uv = source_uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(source, samp, in.uv);
}
