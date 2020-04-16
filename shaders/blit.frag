#version 450

struct V2F {
    vec2 uv;
};

layout(location = 0) in V2F v_in;
layout(location = 0) out vec4 out_color;

layout(binding = 0)
uniform sampler samp;

layout(binding = 1)
uniform texture2D source;

void main() {
    out_color = texture(sampler2D(source, samp), v_in.uv);
}
