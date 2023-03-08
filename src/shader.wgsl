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

fn rayDirection(fieldOfView: f32, xy: vec2<f32>) -> vec3<f32> {
    var z: f32 = 2. / tan(radians(fieldOfView) / 2.0);
    return normalize(vec3<f32>(xy, z));
}

// Fragment shader

@group(0) @binding(0) var t_diffuse: texture_3d<f32>;
@group(0) @binding(1) var s_diffuse: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var dir: vec3<f32> = rayDirection(45.0, in.clip_position.xy);
    var eye: vec3<f32> = vec3<f32>(0.0, 0.0, 5.0);
    var pos: vec3<f32> = eye + dir * 0.5;
    //return vec4<f32>(dir.z, 0., 0.0, 1.0);

    for (var i: i32 = 0; i < 1000; i = i + 1) {
        if (pos.x < -1.0 || pos.x > 1.0 || pos.y < -1.0 || pos.y > 1.0 || pos.z < -1.0 || pos.z > 1.0) {
            pos = pos + dir * 0.01;
            continue;
        }

        var sampling_point: vec3<i32> = vec3<i32>(
            i32((pos.x + 1.) * 4.),
            i32((pos.y + 1.) * 4.),
            i32((pos.z + 1.) * 4.),
        );

        return vec4<f32>(1., 0.0, 0.0, 1.0);

        // var sample: f32 = textureLoad(t_diffuse, sampling_point, 0).r;
        // if (sample > 0.5) {
        //     return vec4<f32>(1.0, 0.0, 0.0, 1.0);
        // }
        // pos = pos + dir * 0.1;
    }

    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}