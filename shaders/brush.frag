#version 450

layout(location = 0) in vec4 v_color;
layout(location = 0) out vec4 out_color;

layout(binding = 2)
uniform texture2D t_Color;

layout(binding = 3)
uniform sampler s_Color;

void main() {
    // out_color = v_color;
    out_color = vec4(1.0, 1.0, 0.0, 1.0);
}
