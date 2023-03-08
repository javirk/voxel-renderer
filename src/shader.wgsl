// Vertex shader

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) tex_pos: vec3<f32>,
};

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.color = model.color;
    out.clip_position = vec4<f32>(model.position, 1.0);
    //out.clip_position = vec4<f32>((model.position.xy - vec2<f32>(1.0,1.0))/2.0, 0., 0.);
    out.tex_pos = (model.position - vec3<f32>(1.0, 1.0, 1.0)) / 2.0;
    return out;
}

// Fragment shader

@group(0) @binding(0) var t_diffuse: texture_3d<f32>;
@group(0) @binding(1) var s_diffuse: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Sample from the texture on position (pos, 4)
    var pos: vec3<i32> = vec3<i32>(
        i32(in.tex_pos.x * 8.0),
        i32(in.tex_pos.y * 8.0),
        3
    );
    
    return textureLoad(t_diffuse, pos, 0);
    //return vec4<f32>(in.tex_pos, 1.0);
}