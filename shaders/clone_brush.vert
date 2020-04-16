#version 450

#define PRIM_BUFFER_LEN 1

layout(std140, binding = 0)
uniform Globals {
    vec2 u_resolution;
    vec2 u_scroll_offset;
    float u_zoom;
};

struct Primitive {
    int dummy;
};

layout(std140, binding = 1)
uniform u_primitives { Primitive primitives[PRIM_BUFFER_LEN]; };

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
    gl_Position = vec4(transformed_pos, z / 1000.0, 1.0);
    
    v_out.color = a_color;//vec4(1.0, 1.0, 1.0, 1.0);
    v_out.uv_background_src = a_uv_background_src;
    v_out.uv_background_target = a_uv_background_target;
    v_out.uv_brush = a_uv_brush;
}
