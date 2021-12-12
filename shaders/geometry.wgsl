struct VertexOutput {
    [[location(0)]] v_color: vec4<f32>;
    [[builtin(position)]] position: vec4<f32>;
};

[[block]]
struct Globals {
    u_resolution: vec2<f32> ;
    u_scroll_offset: vec2<f32>;
    u_zoom: f32;
};

[[block]]
struct Primitive {
    matrix: mat4x4<f32>;
    color: vec4<f32>;
    width: f32;
};

[[group(0), binding(0)]]
var<uniform> globals: Globals;

[[group(0), binding(1)]]
var<uniform> primitives: Primitive;

[[stage(vertex)]]
fn vs_main(
    [[location(0)]] a_position: vec2<f32>,
    [[location(1)]] a_normal: vec2<f32>,
    [[location(2)]] a_prim_id: i32,
    ) -> VertexOutput {
    // int id = a_prim_id;
    let prim = primitives;

    let local_pos: vec2<f32> = a_position + a_normal * prim.width;
    var out: VertexOutput;
    out.position = prim.matrix * vec4<f32>(local_pos, 0.0, 1.0); // - u_scroll_offset + prim.translate + 5.0 * vec2(float(gl_InstanceIndex), 0.0);
    // vec2 transformed_pos = world_pos * u_zoom / (0.5 * u_resolution);
    // Move (0,0) from the center of the screen to the top-left corner
    // transformed_pos -= 1.0;

    // float z = float(prim.z_index) / 4096.0;
    // gl_Position = vec4(transformed_pos, z / 1000.0, 1.0);
    
    //v_color = vec4(mod(gl_VertexIndex, 3) == 0, mod(gl_VertexIndex, 3) == 1, mod(gl_VertexIndex, 3) == 2, 1);
    out.v_color = prim.color;
    return out;
}

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    return in.v_color;
}
