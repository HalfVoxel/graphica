#version 450

struct V2F {
    vec4 color;
    vec2 uv;
};

layout(location = 0) in V2F v_in;
// layout(location = 0) in vec4 v_color;
// layout(location = 4) in vec2 v_uv;
layout(location = 0) out vec4 out_color;

layout(binding = 2)
uniform texture2D t_Color;

layout(binding = 3)
uniform sampler s_Color;

void main() {
    out_color = v_in.color * texture(sampler2D(t_Color, s_Color), v_in.uv);
    // out_color = vec4(1.0, 1.0, 0.0, 1.0);
}
