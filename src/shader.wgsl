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
fn phongIllumination(k_a: vec3<f32>, k_d: vec3<f32>, k_s: vec3<f32>, alpha: f32, p: vec3<f32>, eye: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    var ambientLight: vec3<f32> = 0.5 * vec3<f32>(1.0, 1.0, 1.0);
    var color: vec3<f32> = ambientLight * k_a;
    
    var light1Pos: vec3<f32> = vec3<f32>(2., 0., 5.);
    var light1Intensity: vec3<f32> = vec3<f32>(0.4, 0.4, 0.4);
    
    color += phongContribForLight(k_d, k_s, alpha, p, eye, n, light1Pos, light1Intensity);
    
    return color;
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
            var color = phongIllumination(K_a, K_d, K_s, shininess, pos, lookfrom, normal);
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
        return vec4<f32>(1.0, 0.0, 0.0, 1.0);
    }
    var pos: vec3<f32> = lookfrom + dir * t.x;
    var final_pos: vec3<f32> = lookfrom + dir * t.y;

    var voxel_size: vec3<f32> = 2. / unif.texture_dims;

    var sampling_point: vec3<i32> = vec3<i32>(  // The initial voxel
        i32((pos.x + 1.) * (unif.texture_dims.x / 2. - 1e-3)),
        i32((pos.y + 1.) * (unif.texture_dims.y / 2. - 1e-3)),
        i32((pos.z + 1.) * (unif.texture_dims.z / 2. - 1e-3)),
    );

    var center: vec3<f32> = get_voxel_center(sampling_point);
    var next_bound: vec3<f32> = center + sign(dir) * voxel_size * 1.;
    var step_dir: vec3<i32> = vec3<i32>(sign(dir));
    var t_max: vec3<f32> = (next_bound - pos) / dir;
    var delta_dist: vec3<f32> = voxel_size / (dir);

    var t_count: vec3<f32> = vec3<f32>(0., 0., 0.);
    //return trivial_marching(pos, dir, K_a, K_d, K_s, shininess, lookfrom);

    // DDA algorithm:
    for (var i: i32 = 0; i < 1000; i++) {
        var sample: f32 = textureLoad(t_diffuse, sampling_point, 0).r;
        if (sample > 0.) {
            // Calculate center
            center = get_voxel_center(sampling_point);
            var normal = get_normal(center, pos);            
            var color = phongIllumination(K_a, K_d, K_s, shininess, pos, lookfrom, normal);
            return vec4<f32>(color, 1.0);
            // return vec4<f32>(1., 0., 0., 1.0);
            // return vec4<f32>(f32(sampling_point.x) / 64., f32(sampling_point.y) / 64., f32(sampling_point.z) / 64., 1.);
        }

        if (t_max.x <= t_max.y) {
            if (t_max.x <= t_max.z) {
                // Move in x direction
                t_max.x += delta_dist.x;
                if abs(t_max.x) > 1. {
                    t_count.x += 1.;
                    break;
                } 
                sampling_point.x += step_dir.x;
            } else {
                t_max.z += delta_dist.z;
                if abs(t_max.z) > 1. {
                    t_count.z += 1.;
                    break;
                } 
                sampling_point.z += step_dir.z;
            }
        } else {
            if (t_max.y < t_max.z) {
                t_max.y += delta_dist.y;
                if abs(t_max.y) > 1. {
                    t_count.y += 1.;
                    break
                } 
                sampling_point.y += step_dir.y;
            } else {
                t_max.z += delta_dist.z;
                if abs(t_max.z) > 1. {
                    t_count.z += 1.;
                    break;
                } 
                sampling_point.z += step_dir.z;
            }
        }
        
    }
    return vec4<f32>(0.,0., 0., 1.0);
}