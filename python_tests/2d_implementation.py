import torch
import numpy as np

import matplotlib.pyplot as plt


# AABB intersection
def intersect_box(ray_origin, ray_dir, box_min, box_max):
    tmin = (box_min - ray_origin) / ray_dir
    tmax = (box_max - ray_origin) / ray_dir

    tmin, tmax = torch.minimum(tmin, tmax), torch.maximum(tmin, tmax)
    tmin = torch.max(tmin)
    tmax = torch.min(tmax)

    return tmin, tmax

def fast_voxel_traversal(dir, world, tmin, tmax):
    # Fast voxel traversal algorithm
    # https://www.cse.yorku.ca/~amana/research/grid.pdf
    voxel_size = torch.tensor([2 / world_size, 2 / world_size])
    pos = eye + tmin * dir
    i_voxel = torch.clip(((pos + 1) / 2 * torch.tensor([world_size, world_size])).int(), max=world_size - 1, min=0)
    dir_sign = torch.sign(dir).int()
    bound = (i_voxel + (dir_sign > 0).int()) * voxel_size
    bound = bound * 2 - 1
    # bound = eye + tmax * dir
    inv_dir = 1 / dir
    t = (bound - pos) * inv_dir * dir_sign
    t_delta = voxel_size * inv_dir * dir_sign

    t_total = 0
    all_pos = []

    for i in range(64):
        sampling_point = i_voxel
        if world[sampling_point[0], sampling_point[1]] == 1:
            world[sampling_point[0], sampling_point[1]] = 0.5
            # Return intersection point
            final_pos = pos + t_total * dir
            # return all_pos, world

        if t[0] < t[1]:
            t[0] += t_delta[0]
            i_voxel[0] += dir_sign[0]
            t_total += t_delta[0]
        else:
            t[1] += t_delta[1]
            i_voxel[1] += dir_sign[1]
            t_total += t_delta[1]

        all_pos.append(pos + t_total * dir)

        if torch.any(i_voxel < 0) or torch.any(i_voxel > (world_size - 1)):
            break
    return all_pos, world


def fast_voxel_traversal2(dir, world, tmin, tmax):
    pos = eye + tmin * dir
    voxel_size = torch.tensor([2 / world_size, 2 / world_size])
    unitstepsize = torch.abs(1 / dir) * voxel_size
    vMapCheck = torch.clip(((pos + 1) / 2 * torch.tensor([world_size, world_size])).int(), max=world_size - 1, min=0)
    vStep = torch.sign(dir).int()

    if dir[0] < 0:
        vRayLength1D_x = (pos[0] - vMapCheck[0] * voxel_size[0]) * unitstepsize[0]
    else:
        vRayLength1D_x = (vMapCheck[0] + 1 - pos[0] * voxel_size[0]) * unitstepsize[0]

    if dir[1] < 0:
        vRayLength1D_y = (pos[1] - vMapCheck[1] * voxel_size[1]) * unitstepsize[1]
    else:
        vRayLength1D_y = (vMapCheck[1] + 1 - pos[1] * voxel_size[1]) * unitstepsize[1]

    fDistance = 0
    all_pos = []
    for i in range(64):
        if world[vMapCheck[0], vMapCheck[1]] == 1:
            world[vMapCheck[0], vMapCheck[1]] = 0.5
            # Return intersection point
            all_pos.append(pos + fDistance * dir)
            return all_pos, world

        if vRayLength1D_x < vRayLength1D_y:
            vMapCheck[0] += vStep[0]
            fDistance = vRayLength1D_x
            vRayLength1D_x += unitstepsize[0]
        else:
            vMapCheck[1] += vStep[1]
            fDistance = vRayLength1D_y
            vRayLength1D_y += unitstepsize[1]

        all_pos.append(pos + fDistance * dir)

        if torch.any(vMapCheck < 0) or torch.any(vMapCheck > (world_size - 1)):
            break

    return all_pos, world


# Visualization
class Visualization:
    def __init__(self, resolution):
        self.resolution = resolution
        self.x_min, self.x_max = -2., 2.
        self.y_min, self.y_max = -2., 2.
        # self.plt_world = np.zeros((int((self.y_max - self.y_min) * world_size // 2),
        #                            int((self.x_max - self.x_min) * world_size // 2)))
        # initial_x = int((-1 - self.x_min) / 2 * world_size)
        # initial_y = int((-1 - self.y_min) / 2 * world_size)
        self.plt_world = torch.zeros((resolution, resolution))
        initial_x = int((-1 - self.x_min) * resolution / (self.x_max - self.x_min))
        initial_y = int((-1 - self.y_min) * resolution / (self.y_max - self.y_min))
        height_world = int(2 * resolution / (self.y_max - self.y_min))
        width_world = int(2 * resolution / (self.x_max - self.x_min))
        world_resized = torch.functional.F.interpolate(world.unsqueeze(0).unsqueeze(0), size=(height_world, width_world))

        self.plt_world[initial_x:initial_x + height_world, initial_y:initial_y + width_world] = world_resized
        self.fig, self.ax = plt.subplots(1, tight_layout=True)
        self.converter = lambda x, y: ((x - self.y_min) * resolution / (self.y_max - self.y_min),
                                       (y - self.x_min) * resolution / (self.x_max - self.x_min))

    def add_point_world(self, point, color='r'):
        new_pos = self.converter(point[0], point[1])
        self.ax.plot(new_pos[1], new_pos[0], color + 'o')

    def add_line_world(self, dir, color='r'):
        new_pos = self.converter(eye[0], eye[1])
        self.ax.plot([new_pos[1], new_pos[1] + dir[1] * self.resolution], [new_pos[0], new_pos[0] + dir[0] * self.resolution], color + '-')

    def show(self):
        self.ax.imshow(self.plt_world)
        plt.show()


if __name__ == '__main__':
    eye = torch.tensor([2, 0])
    dir = torch.tensor([-1., -0.5])

    dir = dir / torch.norm(dir)

    world_size = 10

    # world = np.zeros((world_size, world_size, world_size))
    world = torch.randint(0, 2, size=(world_size, world_size), dtype=torch.float)
    # world = torch.ones((world_size, world_size), dtype=torch.float)

    # Find first intersection
    tmin, tmax = intersect_box(eye, dir, torch.tensor([-1, -1]), torch.tensor([1, 1]))
    if tmin > tmax:
        print('No intersection')
        exit(0)
    first_intersection = eye + dir * tmin
    last_intersection = eye + dir * tmax

    # Fast voxel traversal
    positions, world = fast_voxel_traversal2(dir, world, tmin, tmax)

    vis = Visualization(512)
    vis.add_point_world(eye)
    vis.add_line_world(dir)
    vis.add_point_world(first_intersection, 'g')
    vis.add_point_world(last_intersection, 'g')
    for position in positions:
        vis.add_point_world(position, 'b')
    vis.show()
