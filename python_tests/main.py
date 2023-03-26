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
    ray_dir = normalize(ray_dir)
    return ray_dir


# AABB intersection
def intersect_box(ray_origin, ray_dir, box_min, box_max):
    box_min += 1e-6
    box_max -= 1e-6
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
        sampling_point = np.clip(((pos + 1) / 2 * np.array([world_size, world_size, world_size])).astype(int), a_max=world_size - 1, a_min=0)
        if world[sampling_point[0], sampling_point[1], sampling_point[2]] == 1:
            return np.array([1, 1, 1])
        t += 0.05
    return np.zeros(3)


def test_fvt(pos, i_voxel, t):
    new_pos = pos + t * dir
    i_voxel_new = np.clip(((new_pos + 1) / 2 * np.array([world_size, world_size, world_size])).astype(int), a_max=world_size - 1, a_min=0)
    assert np.all(i_voxel_new == i_voxel)


def fast_voxel_traversal(dir, world, tmin, tmax):
    # Fast voxel traversal algorithm
    # https://www.cse.yorku.ca/~amana/research/grid.pdf
    voxel_size = np.array([1 / world_size, 1 / world_size, 1 / world_size])
    pos = eye + tmin * dir
    i_voxel = np.clip(((pos + 1) / 2 * np.array([world_size, world_size, world_size])).astype(int), a_max=world_size - 1, a_min=0)
    dir_sign = np.sign(dir)
    bound = (i_voxel + (dir_sign > 0).astype(int)) * voxel_size
    bound = bound * 2 - 1
    # bound = eye + tmax * dir
    inv_dir = 1 / dir
    t = (bound - pos) * inv_dir #* dir_sign
    t_delta = voxel_size * inv_dir * dir_sign

    t_total = 0

    for i in range(64):
        sampling_point = i_voxel
        if world[sampling_point[0], sampling_point[1], sampling_point[2]] == 1:
            # test_fvt(pos, i_voxel, t_total)
            return np.array([1, 1, 1])

        if t[0] < t[1]:
            if t[0] < t[2]:
                t[0] += t_delta[0]
                i_voxel[0] += dir_sign[0]
                t_total += t_delta[0]
            else:
                t[2] += t_delta[2]
                i_voxel[2] += dir_sign[2]
                t_total += t_delta[2]
        else:
            if t[1] < t[2]:
                t[1] += t_delta[1]
                i_voxel[1] += dir_sign[1]
                t_total += t_delta[1]
            else:
                t[2] += t_delta[2]
                i_voxel[2] += dir_sign[2]
                t_total += t_delta[2]

        if np.any(i_voxel < 0) or np.any(i_voxel > 63):
            break
        # test_fvt(pos, i_voxel, t_total)
    return np.zeros(3)


screen_to_pixel = lambda x, y: (int((x + 1) / 2 * width), int((y + 1) / 2 * height))

eye = np.array([2., 3., 5.])
up = np.array([0., 1., 0.])
lookat = np.array([0., 0., 0.])

width = 200
height = 200
world_size = 64

world = np.zeros((world_size, world_size, world_size))
# Make a sphere into world
for x in range(world_size):
    for y in range(world_size):
        for z in range(world_size):
            if np.linalg.norm(np.array([x, y, z]) - np.array([world_size // 2, world_size // 2, world_size // 2])) < (world_size // 2):
                world[x, y, z] = 1

x_all_pos = np.linspace(-1, 1, num=width, endpoint=False)
y_all_pos = np.linspace(-1, 1, num=height, endpoint=False)

color = np.zeros((width, height, 3))

for x_screen in tqdm(x_all_pos):
    for y_screen in y_all_pos:
        xy_screen = np.array([x_screen, y_screen])
        dir = ray_direction(eye, lookat, up, 45, xy_screen)  # + np.random.normal(0, 0.01, size=3)
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
