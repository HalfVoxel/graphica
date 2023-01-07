struct VertexOutput {
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @builtin(position) position: vec4<f32>,
}

struct PassInfo {
    render_target_size: vec2<f32>,
}


struct Globals {
    u_resolution: vec2<f32>,
    u_scroll_offset: vec2<f32>,
    u_zoom: f32,
}

struct Primitive {
    local2world: mat4x4<f32>,
}

@group(0) @binding(0)
var<uniform> globals: Globals;

@group(0) @binding(1)
var<uniform> primitives: Primitive;

@group(0) @binding(2)
var s_Color: sampler;

@group(0) @binding(3)
var t_Color: texture_2d<f32>;

@group(1) @binding(0)
var<uniform> pass_info: PassInfo;

@vertex
fn vs_main(
    @location(0) a_position: vec2<f32>,
    @location(1) a_uv: vec2<f32>,
    @location(2) a_color: vec4<f32>,
    ) -> VertexOutput {
    let prim: Primitive = primitives;
    var out: VertexOutput;
    let size = vec2<f32>(pass_info.render_target_size);
    var p = (a_position / size) * 2.0 - vec2<f32>(1.0); //prim.matrix * vec4<f32>(a_position, 0.0, 1.0);
    p = vec2<f32>(p.x, -p.y);
    out.position = vec4<f32>(p, 0.0, 1.0);
    // out.position = vec4<f32>(out.position.xy * pass_info.render_target_size.xy, out.position.zw);
    out.color = a_color;//vec4(1.0, 1.0, 1.0, 1.0);
    out.uv = a_uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let out_color = in.color * textureSample(t_Color, s_Color, in.uv);
    return out_color;
}
