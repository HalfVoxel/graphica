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


layout(location = 0) in vec2 a_position;

layout(location = 0) out vec4 v_color;

void main() {
    vec2 local_pos = a_position;
    vec2 world_pos = local_pos - u_scroll_offset;
    vec2 transformed_pos = world_pos * u_zoom / (0.5 * u_resolution);
    // Move (0,0) from the center of the screen to the top-left corner
    // transformed_pos -= 1.0;

    float z = 1.0;
    gl_Position = vec4(transformed_pos, z / 1000.0, 1.0);
    
    v_color = vec4(1.0, 1.0, 1.0, 1.0);
}
