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
    out.clip_position = vec4<f32>(model.position, 1.0);  // [-1, 1]?
    //out.clip_position = vec4<f32>((model.position.xy - vec2<f32>(1.0,1.0))/2.0, 0., 0.);
    out.tex_pos = model.position;
    return out;
}


fn rayDirection(fieldOfView: f32, xy: vec2<f32>) -> vec3<f32> {
    var z: f32 = 2. / tan(radians(fieldOfView) / 2.0);
    return normalize(vec3<f32>(xy, -z));
}

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
        (f32(voxel.x) + 0.5) / 32. - 1.,
        (f32(voxel.y) + 0.5) / 32. - 1.,
        (f32(voxel.z) + 0.5) / 32. - 1.,
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

// Fragment shader

@group(0) @binding(0) var t_diffuse: texture_3d<f32>;
@group(0) @binding(1) var s_diffuse: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    //var xy: vec2<f32> = vec2<f32>(0., 0.);
    var dir: vec3<f32> = rayDirection(45.0, in.tex_pos.xy);
    var eye: vec3<f32> = vec3<f32>(0.0, 0.0, 10.0);
    
    // We start outside of the volume, so we need to find the first intersection. If there is none, the pixel is black.
    var t: vec2<f32> = intersectAABB(eye, dir, vec3<f32>(-1.0, -1.0, -1.0), vec3<f32>(1.0, 1.0, 1.0));
    if (t.x > t.y) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    var pos: vec3<f32> = eye + dir * t.x;

    var sampling_point: vec3<i32> = vec3<i32>(  // The initial voxel
        i32((pos.x + 1.) * 32.),
        i32((pos.y + 1.) * 32.),
        i32((pos.z + 1.) * 32.),
    );

    var map_pos: vec3<i32> = vec3<i32>(floor(pos));
    var delta_dist: vec3<f32> = abs(vec3<f32>(length(dir)) / dir);
    //var ray_step: vec3<i32> = vec3<i32>(sign(dir));
    var side_dist: vec3<f32> = (sign(dir) * (vec3<f32>(sampling_point) / 32. - pos) + (sign(dir) * 0.5) + 0.5) * delta_dist;
    

    // DDA algorithm:
    for (var i: i32 = 0; i < 64; i++) {
        var sample: f32 = textureLoad(t_diffuse, sampling_point, 0).r;
        if (sample > 0.5) {
            // Calculate center
            var center: vec3<f32> = get_voxel_center(sampling_point);
            var normal = get_normal(center, pos);
            var K_a: vec3<f32> = vec3<f32>(0.2, 0.2, 0.2);
            var K_d: vec3<f32> = vec3<f32>(0.7, 0.2, 0.2);
            var K_s: vec3<f32> = vec3<f32>(1.0, 1.0, 1.0);
            var shininess: f32 = 10.0;
            
            var color = phongIllumination(K_a, K_d, K_s, shininess, pos, eye, normal);
            return vec4<f32>(color, 1.0);
        }

        if (side_dist.x < side_dist.y) {
            if (side_dist.x < side_dist.z) {
                // Move in x direction
                side_dist.x += delta_dist.x;
                sampling_point.x += 1;
            }
            else {
                side_dist.z += delta_dist.z;
                sampling_point.z += 1;
            }
        }
        else {
            if (side_dist.y < side_dist.z) {
                side_dist.y += delta_dist.y;
                sampling_point.y += 1;
            }
            else {
                side_dist.z += delta_dist.z;
                sampling_point.z += 1;
            }
        }
        
    }
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);


    // for (var i: i32 = 0; i < 2500; i = i + 1) {
    //     // var t: vec2<f32> = intersectAABB(pos, dir, vec3<f32>(-1.0, -1.0, -1.0), vec3<f32>(1.0, 1.0, 1.0));
    //     if (pos.x < -1.0 || pos.x > 1.0 || pos.y < -1.0 || pos.y > 1.0 || pos.z < -1.0 || pos.z > 1.0) {
    //         pos = pos + dir * 0.005;
    //         continue;
    //     } 

    //     var sampling_point: vec3<i32> = vec3<i32>(
    //         i32((pos.x + 1.) * 32.),
    //         i32((pos.y + 1.) * 32.),
    //         i32((pos.z + 1.) * 32.),
    //     );

    //     var sample: f32 = textureLoad(t_diffuse, sampling_point, 0).r;
    //     if (sample > 0.5) {
    //         // Calculate center
    //         var center: vec3<f32> = get_voxel_center(sampling_point);
    //         var normal = get_normal(center, pos);
    //         //return vec4<f32>(normal, 1.0);
    //         //return vec4<f32>(1.0, 0.0, 0.0, 1.0);
    //         var K_a: vec3<f32> = vec3<f32>(0.2, 0.2, 0.2);
    //         var K_d: vec3<f32> = vec3<f32>(0.7, 0.2, 0.2);
    //         var K_s: vec3<f32> = vec3<f32>(1.0, 1.0, 1.0);
    //         var shininess: f32 = 10.0;
            
    //         var color = phongIllumination(K_a, K_d, K_s, shininess, pos, eye, normal);
    //         // Translate the previous lines to WGSL and output the color
    //         return vec4<f32>(color, 1.0);
    //     }
    //     pos = pos + dir * 0.005;
    // }

    // return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}