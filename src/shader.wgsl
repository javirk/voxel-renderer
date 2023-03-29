// Vertex shader

let TEXTURE_DIMENSION: f32 = 64.;  // TODO: Make it come from a uniform

struct Uniforms {
    iTime: f32,
    texture_dims: vec3<f32>
}

struct CameraUniforms {
    eye: vec3<f32>,
    lookat: vec3<f32>,
}

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
    out.clip_position = vec4<f32>(model.position, 1.0);  // [-1, 1]?
    //out.clip_position = vec4<f32>((model.position.xy - vec2<f32>(1.0,1.0))/2.0, 0., 0.);
    out.tex_pos = model.position;
    return out;
}


// Fragment shader

@group(0) @binding(0) var<uniform> unif: Uniforms;
@group(1) @binding(0) var t_diffuse: texture_3d<f32>;
@group(1) @binding(1) var s_diffuse: sampler;
@group(2) @binding(0) var<uniform> camera: CameraUniforms;


fn get_normal(center: vec3<f32>, hit_point: vec3<f32>) -> vec3<f32> {
    var distance_vec: vec3<f32> = hit_point - center;
    var vector_positive = abs(distance_vec);
    if (vector_positive.x > vector_positive.y) && (vector_positive.x > vector_positive.z) {
        return vec3<f32>(sign(distance_vec.x), 0., 0.);
    } else if (vector_positive.y > vector_positive.x) && (vector_positive.y > vector_positive.z) {
        return vec3<f32>(0., sign(distance_vec.y), 0.);
    } else {
        return vec3<f32>(0., 0., sign(distance_vec.z));
    }
}


fn get_voxel_center(voxel: vec3<i32>) -> vec3<f32> {
    return vec3<f32>(
        (f32(voxel.x) + 0.5) / (unif.texture_dims.x / 2.) - 1.,
        (f32(voxel.y) + 0.5) / (unif.texture_dims.y / 2.) - 1.,
        (f32(voxel.z) + 0.5) / (unif.texture_dims.z / 2.) - 1.,
    );
}


fn intersectAABB(rayOrigin: vec3<f32>, rayDir: vec3<f32>, boxMin: vec3<f32>, boxMax: vec3<f32>) -> vec2<f32> {
    // compute the near and far intersections of the cube (stored in the x and y components) using the slab method
    // no intersection means vec.x > vec.y (really tNear > tFar)    
    var tMin: vec3<f32> = (boxMin - rayOrigin) / rayDir;
    var tMax: vec3<f32> = (boxMax - rayOrigin) / rayDir;
    var t1: vec3<f32> = min(tMin, tMax);
    var t2: vec3<f32> = max(tMin, tMax);
    var tNear: f32 = max(max(t1.x, t1.y), t1.z);
    var tFar: f32 = min(min(t2.x, t2.y), t2.z);
    return vec2<f32>(tNear, tFar);
}

fn color_map(value: f32) -> vec3<f32> {
    var color: vec3<f32> = vec3<f32>(0., 0., 0.);
    /*
        Light Pink: RGB (1.000, 0.714, 0.757)
        Light Blue: RGB (0.678, 0.847, 0.902)
        Khaki: RGB (0.941, 0.902, 0.549)
        Light Green: RGB (0.565, 0.933, 0.565)
        Light Gray: RGB (0.827, 0.827, 0.827)
        Peach: RGB (1.000, 0.855, 0.725)
        Light Steel Blue: RGB (0.690, 0.769, 0.871)
        Wheat: RGB (0.961, 0.871, 0.702)
        Misty Rose: RGB (1.000, 0.894, 0.882)
        Silver: RGB (0.753, 0.753, 0.753)
    */
    if (value < 0.1) {
        color = vec3<f32>(1.000, 0.714, 0.757);
    } else if (value < 0.2) {
        color = vec3<f32>(0.678, 0.847, 0.902);
    } else if (value < 0.3) {
        color = vec3<f32> (0.941, 0.902, 0.549);
    } else if (value < 0.4) {
        color = vec3<f32>(0.565, 0.933, 0.565);
    } else if (value < 0.5) {
        color = vec3<f32> (0.827, 0.827, 0.827);
    } else if (value < 0.6) {
        color = vec3<f32>(1.000, 0.855, 0.725);
    } else if (value < 0.7) {
        color = vec3<f32>(0.690, 0.769, 0.871);
    } else if (value < 0.8) {
        color = vec3<f32> (0.961, 0.871, 0.702);
    } else if (value < 0.9) {
        color = vec3<f32>(1.000, 0.894, 0.882);
    } else {
        color = vec3<f32>(0.753, 0.753, 0.753);
    }
    return color;
}

/**
 * Lighting contribution of a single point light source via Phong illumination.
 * 
 * The vec3 returned is the RGB color of the light's contribution.
 *
 * k_a: Ambient color
 * k_d: Diffuse color
 * k_s: Specular color
 * alpha: Shininess coefficient
 * p: position of point being lit
 * eye: the position of the camera
 * lightPos: the position of the light
 * lightIntensity: color/intensity of the light
 *
 * See https://en.wikipedia.org/wiki/Phong_reflection_model#Description
 */
fn phongContribForLight(k_d: vec3<f32>, k_s: vec3<f32>, alpha: f32, p: vec3<f32>, eye: vec3<f32>, n: vec3<f32>,
                          lightPos: vec3<f32>, lightIntensity: vec3<f32>) -> vec3<f32> {
    var N = n;
    var L = normalize(lightPos - p);
    var V = normalize(eye - p);
    var R = normalize(reflect(-L, N));
    
    var dotLN = dot(L, N);
    var dotRV = dot(R, V);
    
    if (dotLN < 0.0) {
        // Light not visible from this point on the surface
        return vec3<f32>(0.0, 0.0, 0.0);
    } 
    
    if (dotRV < 0.0) {
        // Light reflection in opposite direction as viewer, apply only diffuse
        // component
        return lightIntensity * (k_d * dotLN);
    }
    return lightIntensity * (k_d * dotLN + k_s * pow(dotRV, alpha));
}

/**
 * Lighting via Phong illumination.
 * 
 * The vec3 returned is the RGB color of that point after lighting is applied.
 * k_a: Ambient color
 * k_d: Diffuse color
 * k_s: Specular color
 * alpha: Shininess coefficient
 * p: position of point being lit
 * eye: the position of the camera
 *
 * See https://en.wikipedia.org/wiki/Phong_reflection_model#Description
 */
fn phongIllumination(k_a: vec3<f32>, k_d: vec3<f32>, k_s: vec3<f32>, alpha: f32, p: vec3<f32>, eye: vec3<f32>, n: vec3<f32>, object_color: vec3<f32>) -> vec3<f32> {
    var ambientLight: vec3<f32> = 0.1 * vec3<f32>(1.0, 1.0, 1.0);
    var color: vec3<f32> = ambientLight * k_a;
    
    var light1Pos: vec3<f32> = vec3<f32>(2., 0., 5.);
    var light1Intensity: vec3<f32> = 0.5*vec3<f32>(1.0, 1.0, 0.9);
    
    color += phongContribForLight(k_d, k_s, alpha, p, eye, n, light1Pos, light1Intensity);

    var light2Pos: vec3<f32> = vec3<f32>(-2., 2., -5.);
    var light2Intensity: vec3<f32> = 0.5*vec3<f32>(1.0, 1.0, 0.9);
    
    color += phongContribForLight(k_d, k_s, alpha, p, eye, n, light2Pos, light2Intensity);
    
    return color * object_color;
}

fn rayDirection(lookfrom: vec3<f32>, lookat: vec3<f32>, up: vec3<f32>, fov: f32, tex_pos: vec2<f32>) -> vec3<f32> {
    var tex_pos_norm = (tex_pos + 1.) / 2.;  // TODO: Please... this is so ugly. Change the other code to use normalized coordinates (-1, 1).
    var aspect_ratio: f32 = 600. / 800.;
    var theta: f32 = radians(fov);
    var half_width: f32 = tan(theta / 2.);
    var half_height: f32 = half_width * aspect_ratio;

    var w: vec3<f32> = normalize(lookfrom - lookat);
    var u: vec3<f32> = normalize(cross(up, w));
    var v: vec3<f32> = cross(w, u);

    var lower_left_corner: vec3<f32> = lookfrom - half_width * u - half_height * v - w;
    var horizontal: vec3<f32> = 2. * half_width * u;
    var vertical: vec3<f32> = 2. * half_height * v;
    var ray_dir: vec3<f32> = lower_left_corner + tex_pos_norm.x * horizontal + tex_pos_norm.y * vertical - lookfrom;
    return ray_dir;
}

fn trivial_marching(pos: vec3<f32>, dir: vec3<f32>, K_a: vec3<f32>, K_d: vec3<f32>, K_s: vec3<f32>, shininess: f32, lookfrom: vec3<f32>) -> vec4<f32> {
    var pos: vec3<f32> = pos;
    for (var i: i32 = 0; i < 20000; i++) {
        
        var sampling_point: vec3<i32> = vec3<i32>(
            i32((pos.x + 1.) * (unif.texture_dims.x / 2. - 1e-3)),
            i32((pos.y + 1.) * (unif.texture_dims.y / 2. - 1e-3)),
            i32((pos.z + 1.) * (unif.texture_dims.z / 2. - 1e-3)),
        );
        var sample: f32 = textureLoad(t_diffuse, sampling_point, 0).r;
        if (sample > 0.) {
            var center = get_voxel_center(sampling_point);
            var normal = get_normal(center, pos);
            var object_color = color_map(sample);        
            var color = phongIllumination(K_a, K_d, K_s, shininess, pos, lookfrom, normal, object_color);
            //color = vec3<f32>(1., 0., 0.);
            return vec4<f32>(color, 1.0);
        }

        pos = pos + dir * 0.0001;
        // Check if the position is out of the borders
        if (pos.x < -1. || pos.x > 1. || pos.y < -1. || pos.y > 1. || pos.z < -1. || pos.z > 1.) {
            return vec4<f32>(0., 0., 0., 1.);
        }
    }
    return vec4<f32>(0.,0., 0., 1.0);
}

fn fast_marching(pos: vec3<f32>, dir: vec3<f32>, K_a: vec3<f32>, K_d: vec3<f32>, K_s: vec3<f32>, shininess: f32, lookfrom: vec3<f32>) -> vec4<f32> {
    var voxel_size = 2./ unif.texture_dims;
    var i_voxel: vec3<i32> = clamp(vec3<i32>((pos + 1.) / 2. * unif.texture_dims), vec3(0, 0, 0), vec3<i32>(unif.texture_dims) - 1);
    var dir_sign: vec3<f32> = sign(dir);
    var unitstepsize: vec3<f32> = voxel_size * dir_sign / dir;
    var bound: vec3<f32> = (vec3<f32>(i_voxel) + clamp(dir_sign, vec3(0., 0., 0.), vec3(1., 1., 1.))) / unif.texture_dims;
    bound = bound * 2. - 1.;
    var ray_length = abs(bound - pos) * unitstepsize / voxel_size;

    var distance: f32 = 0.;

    for(var i: i32 = 0; i < 128; i++) {
        var sample: f32 = textureLoad(t_diffuse, i_voxel, 0).r;
        if (sample > 0.) {
            var new_pos = pos + dir * distance;
            var center = get_voxel_center(i_voxel);
            var normal = get_normal(center, new_pos);
            var object_color = color_map(sample);  
            var color = phongIllumination(K_a, K_d, K_s, shininess, new_pos, lookfrom, normal, object_color);
            return vec4<f32>(color, 1.0);
        }

        if ray_length.x < ray_length.y {
            if ray_length.x < ray_length.z {
                i_voxel.x += i32(dir_sign.x);
                distance = ray_length.x;
                ray_length.x += unitstepsize.x;
            } else {
                i_voxel.z += i32(dir_sign.z);
                distance = ray_length.z;
                ray_length.z += unitstepsize.z;
            }
        } else {
            if ray_length.y < ray_length.z {
                i_voxel.y += i32(dir_sign.y);
                distance = ray_length.y;
                ray_length.y += unitstepsize.y;
            } else {
                i_voxel.z += i32(dir_sign.z);
                distance = ray_length.z;
                ray_length.z += unitstepsize.z;
            }
        }

        if (i_voxel.x < 0 || i_voxel.x >= i32(unif.texture_dims.x) || i_voxel.y < 0 || i_voxel.y >= i32(unif.texture_dims.y) || i_voxel.z < 0 || i_voxel.z >= i32(unif.texture_dims.z)) {
            return vec4<f32>(1., 1., 1., 1.);
        }
    }
    return vec4<f32>(1., 1., 1., 1.);
}


@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var lookat: vec3<f32> = camera.lookat;
    var lookfrom = camera.eye;
    var up: vec3<f32> = vec3<f32>(0.0, 1.0, 0.0);
    var dir: vec3<f32> = rayDirection(lookfrom, lookat, up, 60.0, in.tex_pos.xy);

    // TODO: Put all this into a light struct
    var K_a: vec3<f32> = vec3<f32>(0.2, 0.2, 0.2);
    var K_d: vec3<f32> = vec3<f32>(0.7, 0.2, 0.2);
    var K_s: vec3<f32> = vec3<f32>(1.0, 1.0, 1.0);
    var shininess: f32 = 10.0;
    
    // We start outside of the volume, so we need to find the first intersection. If there is none, the pixel is black.
    var t: vec2<f32> = intersectAABB(lookfrom, dir, vec3<f32>(-1.0, -1.0, -1.0), vec3<f32>(1.0, 1.0, 1.0));
    if (t.x > t.y) {
        return vec4<f32>(1.0, 1.0, 1.0, 1.0);
    }
    var pos: vec3<f32> = lookfrom + dir * t.x;
    var bound: vec3<f32> = lookfrom + dir * t.y;

    return fast_marching(pos, dir, K_a, K_d, K_s, shininess, lookfrom);
    // return trivial_marching(pos, dir, K_a, K_d, K_s, shininess, lookfrom);
}