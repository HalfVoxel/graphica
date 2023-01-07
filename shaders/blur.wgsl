
// @group(0) @binding(0)
// var samp: sampler;

@group(0) @binding(0)
var source: [[access(read)]] texture_storage_2d<rgba8unorm>;

@group(0) @binding(1)
var target: [[access(write)]] texture_storage_2d<rgba8unorm>;

@group(0) @binding(2)
var weights: [[access(read)]] texture_storage_1d<r32float>;

[[stage(compute), workgroup_size(32, 32, 1)]]
fn cs_main(
    [[builtin(workgroup_id)]] workgroup_id: vec3<i32>,
    [[builtin(workgroup_size)]] workgroup_size: vec3<i32>
) {
    let px = vec2<i32>(workgroup_id.xy*vec2<i32>(32, 32) + workgroup_size.xy);

    var sum: vec4<f32> = vec4<f32>(0.0);
    for (var i: i32 = 0; i < 8; i = i + 1) {
        let w = textureLoad(weights, i);
        let a = textureLoad(source, px + vec2<i32>(i - 4, 0));
        sum = sum + a * w;
    }
    textureStore(target, px, sum);
}

