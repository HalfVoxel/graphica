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
    vec4 color;
    // vec2 translate;
    // int z_index;
    float width;
};

layout(std140, binding = 1)
uniform u_primitives { Primitive primitives[PRIM_BUFFER_LEN]; };

layout(location = 0) in vec2 a_position;
layout(location = 1) in vec2 a_normal;
layout(location = 2) in int a_prim_id;

layout(location = 0) out vec4 v_color;

void main() {
    // int id = a_prim_id;
    Primitive prim = primitives[0];

    vec2 local_pos = a_position + a_normal * prim.width;
    gl_Position = prim.matrix * vec4(local_pos, 0, 1); // - u_scroll_offset + prim.translate + 5.0 * vec2(float(gl_InstanceIndex), 0.0);
    // vec2 transformed_pos = world_pos * u_zoom / (0.5 * u_resolution);
    // Move (0,0) from the center of the screen to the top-left corner
    // transformed_pos -= 1.0;

    // float z = float(prim.z_index) / 4096.0;
    // gl_Position = vec4(transformed_pos, z / 1000.0, 1.0);
    
    //v_color = vec4(mod(gl_VertexIndex, 3) == 0, mod(gl_VertexIndex, 3) == 1, mod(gl_VertexIndex, 3) == 2, 1);
    v_color = prim.color;
}
