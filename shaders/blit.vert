#version 450

struct V2F {
    vec2 uv;
};

layout(location = 0) in vec2 source_uv;
layout(location = 3) in vec2 target_uv;

layout(location = 0) out V2F v_out;

void main() {
    float z = 1.0;
    vec2 transformed_pos = target_uv * 2 - 1;
    gl_Position = vec4(transformed_pos, z / 1000.0, 1.0);
    v_out.uv = source_uv;
}
