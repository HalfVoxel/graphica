#version 450

struct V2F {
    vec4 color;
    vec2 uv_background_src;
    vec2 uv_background_target;
    vec2 uv_brush;
};

layout(location = 0) in vec2 a_position;
layout(location = 1) in vec2 a_uv_background_src;
layout(location = 2) in vec2 a_uv_background_target;
layout(location = 3) in vec2 a_uv_brush;
layout(location = 4) in vec4 a_color;

layout(location = 0) out V2F v_out;
// layout(location = 4) out vec2 v_uv;


void main() {
    float z = 1.0;
    vec2 transformed_pos = a_uv_brush * 2 - 1;
    // transformed_pos.y = -transformed_pos.y;
    gl_Position = vec4(transformed_pos, z / 1000.0, 1.0);
    
    v_out.color = a_color;//vec4(1.0, 1.0, 1.0, 1.0);
    v_out.uv_background_src = a_uv_background_src;
    v_out.uv_background_target = a_uv_background_target;
    v_out.uv_brush = a_uv_brush;
}
