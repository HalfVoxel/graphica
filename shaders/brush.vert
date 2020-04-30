#version 450

#define PRIM_BUFFER_LEN 1

layout(std140, binding = 0)
uniform Globals {
    vec2 u_resolution;
    vec2 u_scroll_offset;
    float u_zoom;
};

struct Primitive {
    mat4x4 matrix;
};

layout(std140, binding = 1)
uniform u_primitives { Primitive primitives[PRIM_BUFFER_LEN]; };

struct V2F {
    vec4 color;
    vec2 uv;
};

layout(location = 0) in vec2 a_position;
layout(location = 1) in vec2 a_uv;
layout(location = 2) in vec4 a_color;

layout(location = 0) out V2F v_out;
// layout(location = 4) out vec2 v_uv;

void main() {
    Primitive prim = primitives[0];
    gl_Position = prim.matrix * vec4(a_position, 0, 1);
    
    v_out.color = a_color;//vec4(1.0, 1.0, 1.0, 1.0);
    v_out.uv = a_uv;
}
