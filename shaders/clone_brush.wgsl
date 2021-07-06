struct VertexOutput {
    [[location(0)]] color: vec4<f32>;
    [[location(1)]] uv_background_src: vec2<f32>;
    [[location(2)]] uv_background_target: vec2<f32>;
    [[location(3)]] uv_brush: vec2<f32>;
    [[builtin(position)]] position: vec4<f32>;
};

[[group(0), binding(0)]]
var samp: sampler;

[[group(0), binding(1)]]
var t_background: texture_2d<f32>;

[[group(0), binding(2)]]
var t_brush: texture_2d<f32>;

[[stage(vertex)]]
fn vs_main(
    [[location(0)]] a_position: vec2<f32>,
    [[location(1)]] a_uv_background_src: vec2<f32>,
    [[location(2)]] a_uv_background_target: vec2<f32>,
    [[location(3)]] a_uv_brush: vec2<f32>,
    [[location(4)]] a_color: vec4<f32>,
    ) -> VertexOutput {

    let z = 1.0;
    var uv_brush: vec2<f32> = a_uv_brush;
    uv_brush.y = 1.0 - uv_brush.y;
    let transformed_pos: vec2<f32> = uv_brush * 2.0 - 1.0;
    var uv_src: vec2<f32> = a_uv_background_src;
    var uv_dst: vec2<f32> = a_uv_background_target;
    uv_src.y = 1.0 - uv_src.y;
    uv_dst.y = 1.0 - uv_dst.y;
    // transformed_pos.y = -transformed_pos.y;
    var v_out: VertexOutput;
    v_out.position = vec4<f32>(transformed_pos, z / 1000.0, 1.0);
    v_out.color = a_color;//vec4(1.0, 1.0, 1.0, 1.0);
    v_out.uv_background_src = uv_src;
    v_out.uv_background_target = uv_dst;
    v_out.uv_brush = uv_brush;
    return v_out;
}

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    let background_src: vec4<f32> = textureSample(t_background, samp, in.uv_background_src);
    let background_target: vec4<f32> = textureSample(t_background, samp, in.uv_background_target);
    let brush: f32 = textureSample(t_brush, samp, in.uv_brush).a;
    let v: vec4<f32> = mix(background_target, background_src, vec4<f32>(brush));
    // out_color = mix(v, vec4(1.0, 0.0, 0.0, 1.0), 0.1);
    return v;
    // out_color = vec4(1.0, 1.0, 0.0, 1.0);
}
