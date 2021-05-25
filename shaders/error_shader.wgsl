struct VertexOutput {
    [[location(0)]] a: vec2<f32>;
    // Logs "expected integer", but it clearly is an integer
    [[location(-1)]] b: vec2<f32>;
    // Logs "expected integer", but it clearly is an integer
    [[location(42323232323232)]] c: vec2<f32>;
    [[location(3)]] v_zoom: f32;
    // Error::UnknownBuiltin
    [[builtin(positionx)]] position: vec4<f32>;
    // Error::UnknownAttribute
    [[locationx(3)]] v_zoom: f32;
    
};

// Error::UnknownConservativeDepth
[[early_depth_test(gt)]]
struct Whatever {
}

[[block]]
struct Globals {
    u_resolution: vec2<f32>;
    u_scroll_offset: vec2<f32>;
    u_zoom: f32;
};

[[block]]
struct Globals {
    /// Error::ZeroSizeOrAlign
    [[align(0)]]
    u_resolution: vec2<f32>;
};

[[group(0), binding(0)]]
var globals: Globals;

[[stage(vertex)]]
fn vs_main(
    [[location(0)]] a_position: vec2<f32>
    ) -> VertexOutput {

    var out: VertexOutput;
    out.position = vec4<f32>(a_position, 0.0000001, 1.0);
    out.v_position = a_position;
    out.v_resolution = globals.u_resolution;
    out.v_scroll_offset = globals.u_scroll_offset;
    out.v_zoom = globals.u_zoom;
    return out;
}

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    // Error::UnknownStorageClass
    let px_position: ptr<f32> = 0.0;

    // Error:: ZeroStride
    let px_position: [[stride(0)]] vec4<f32> = 0.0;

    // Error::UnknownStorageFormat
    let px_position: texture_storage_1d<f32> = 0.0;

    // Error:: UnknownType
    let px_position: vecx4<f32> = 0.0;

    let c = 5.0f33;
    let a: vec4<f32> = textureLoad(in, in);
    let b: vec4<f32> = vec4<f32>(in);

    // Error::InvalidForInitializer
    for (break; i < 5; i = i + 2) {
    }

    return out_color;
}

// Error::UnknownShaderStage
[[stage(fragmentx)]]
fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
}