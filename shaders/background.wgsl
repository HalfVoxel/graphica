struct VertexOutput {
    @location(0) v_position: vec2<f32>,
    @location(1) v_resolution: vec2<f32>,
    @location(2) v_scroll_offset: vec2<f32>,
    @location(3) v_zoom: f32,
    @builtin(position) position: vec4<f32>,
};

struct Globals {
    u_resolution: vec2<f32>,
    u_scroll_offset: vec2<f32>,
    u_zoom: f32,
};

@group(0) @binding(0)
var<uniform> globals: Globals;

@vertex
fn vs_main(
    @location(0) a_position: vec2<f32>
    ) -> VertexOutput {

    var out: VertexOutput;
    let view_pos = a_position * 2.0 - 1.0;
    out.position = vec4<f32>(view_pos, 0.0000001, 1.0);
    out.v_position = view_pos;
    out.v_resolution = globals.u_resolution;
    out.v_scroll_offset = globals.u_scroll_offset;
    out.v_zoom = globals.u_zoom;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Multiplication by 0.5 because v_position goes from -1 to +1.
    let px_position: vec2<f32> = in.v_position * in.v_resolution * 0.5;

    // #005fa4
    let vignette: f32 = clamp(0.7 * length(in.v_position), 0.0, 1.0);
    var out_color: vec4<f32> = mix(
        vec4<f32>(0.0, 0.47, 0.9, 1.0),
        vec4<f32>(0.0, 0.1, 0.64, 1.0),
        vec4<f32>(vignette)
    );

    // TODO: properly adapt the grid while zooming in and out.
    var grid_scale: f32 = 5.0;
    if (in.v_zoom < 2.5) {
        grid_scale = 1.0;
    }

    let pos: vec2<f32> = px_position + in.v_scroll_offset * in.v_zoom;

    if (pos.x % (20.0 / grid_scale * in.v_zoom) <= 1.0 ||
        pos.y % (20.0 / grid_scale * in.v_zoom) <= 1.0) {
        out_color = out_color * 1.2;
    }

    if (pos.x % (100.0 / grid_scale * in.v_zoom) <= 2.0 ||
        pos.y % (100.0 / grid_scale * in.v_zoom) <= 2.0) {
        out_color = out_color * 1.2;
    }

    out_color.a = 1.0;

    return out_color;
}
