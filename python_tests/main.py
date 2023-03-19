import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

normalize = lambda x: x / np.linalg.norm(x)


def ray_direction(lookfrom, lookat, up, fov, position):
    pos_norm = (position + 1.) / 2.
    aspect_ratio = height / width
    theta = fov * np.pi / 180
    half_width = np.tan(theta / 2)
    half_height = half_width * aspect_ratio

    w = normalize(lookfrom - lookat)
    u = normalize(np.cross(up, w))
    v = np.cross(w, u)

    lower_left_corner = lookfrom - half_width * u - half_height * v - w
    horizontal = 2 * half_width * u
    vertical = 2 * half_height * v
    ray_dir = lower_left_corner + pos_norm[0] * horizontal + pos_norm[1] * vertical - lookfrom
    return ray_dir


# AABB intersection
def intersect_box(ray_origin, ray_dir, box_min, box_max):
    tmin = (box_min - ray_origin) / ray_dir
    tmax = (box_max - ray_origin) / ray_dir

    tmin, tmax = np.minimum(tmin, tmax), np.maximum(tmin, tmax)
    tmin = np.max(tmin)
    tmax = np.min(tmax)

    return tmin, tmax

def trivial_marching(dir, world, tmin, tmax):
    t = tmin
    while t < tmax:
        pos = eye + t * dir
        sampling_point = np.clip(((pos + 1) / 2 * np.array([64, 64, 64])).astype(int), a_max=63, a_min=0)
        if world[sampling_point[0], sampling_point[1], sampling_point[2]] == 1:
            return np.array([1, 1, 1])
        t += 0.05
    return np.zeros(3)

def fast_voxel_traversal(dir, world, tmin, tmax):
    # Fast voxel traversal algorithm
    # https://www.cse.yorku.ca/~amana/research/grid.pdf
    voxel_size = np.array([1 / 64, 1 / 64, 1 / 64])
    pos = eye + tmin * dir
    i_voxel = np.clip(((pos + 1) / 2 * np.array([64, 64, 64])).astype(int), a_max=63, a_min=0)
    dir_sign = np.sign(dir)
    bound = (i_voxel + (dir_sign > 0).astype(int)) * voxel_size
    bound = bound * 2 - 1
    inv_dir = 1 / dir
    t = (bound - pos) * inv_dir
    t_delta = voxel_size * inv_dir * dir_sign

    for i in range(64):
        sampling_point = i_voxel
        if world[sampling_point[0], sampling_point[1], sampling_point[2]] == 1:
            return np.array([1, 1, 1])

        if t[0] < t[1]:
            if t[0] < t[2]:
                t[0] += t_delta[0]
                i_voxel[0] += dir_sign[0]
            else:
                t[2] += t_delta[2]
                i_voxel[2] += dir_sign[2]
        else:
            if t[1] < t[2]:
                t[1] += t_delta[1]
                i_voxel[1] += dir_sign[1]
            else:
                t[2] += t_delta[2]
                i_voxel[2] += dir_sign[2]

        if np.any(i_voxel < 0) or np.any(i_voxel > 63):
            break
    return np.zeros(3)



screen_to_pixel = lambda x, y: (int((x + 1) / 2 * width), int((y + 1) / 2 * height))

eye = np.array([2., 3., 5.])
up = np.array([0., 1., 0.])
lookat = np.array([0., 0., 0.])

width = 200
height = 200

world = np.zeros((64, 64, 64))
# Make a sphere into world
for x in range(64):
    for y in range(64):
        for z in range(64):
            if np.linalg.norm(np.array([x, y, z]) - np.array([32, 32, 32])) < 32:
                world[x, y, z] = 1

x_all_pos = np.linspace(-1, 1, num=width, endpoint=False)
y_all_pos = np.linspace(-1, 1, num=height, endpoint=False)

color = np.zeros((width, height, 3))

for x_screen in tqdm(x_all_pos):
    for y_screen in y_all_pos:
        xy_screen = np.array([x_screen, y_screen])
        dir = ray_direction(eye, lookat, up, 45, xy_screen) #+ np.random.normal(0, 0.01, size=3)
        tmin, tmax = intersect_box(eye, dir, np.array([-1., -1, -1]), np.array([1., 1, 1]))
        if tmin > tmax:
            pixel_coords = screen_to_pixel(x_screen, y_screen)
            color[pixel_coords] = np.array([0, 0, 0])
            continue

        # pixel_color = trivial_marching(dir, world, tmin, tmax)
        pixel_color = fast_voxel_traversal(dir, world, tmin, tmax)
        pixel_coords = screen_to_pixel(x_screen, y_screen)
        color[pixel_coords] = pixel_color


plt.imshow(color)
plt.show()
