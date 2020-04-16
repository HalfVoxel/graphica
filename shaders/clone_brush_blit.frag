#version 450

struct V2F {
    vec2 uv;
};

layout(location = 0) in V2F v_in;
// layout(location = 0) in vec4 v_color;
// layout(location = 4) in vec2 v_uv;
layout(location = 0) out vec4 out_color;

layout(binding = 2)
uniform sampler samp;

layout(binding = 3)
uniform texture2D t_background;

layout(binding = 4)
uniform texture2D t_brush;

void main() {
    vec4 color = texture(sampler2D(t_background, samp), v_in.uv);
    out_color = color;
}
