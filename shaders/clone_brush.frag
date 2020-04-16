#version 450

struct V2F {
    vec4 color;
    vec2 uv_background_src;
    vec2 uv_background_target;
    vec2 uv_brush;
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
    vec4 background_src = texture(sampler2D(t_background, samp), v_in.uv_background_src);
    vec4 background_target = texture(sampler2D(t_background, samp), v_in.uv_background_target);
    float brush = texture(sampler2D(t_brush, samp), v_in.uv_brush).a;
    vec4 v = mix(background_target, background_src, brush);
    out_color = v;
    // out_color = vec4(1.0, 1.0, 0.0, 1.0);
}
