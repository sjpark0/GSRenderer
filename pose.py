import numpy as np
import os
import torch
import math
from typing import NamedTuple

class ViewInfo(NamedTuple):
    uid : int
    height : int    
    width : int
    FovY: float
    FovX: float
    world_view_transform: torch.Tensor
    full_proj_transform: torch.Tensor
    camera_center: torch.Tensor

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    #center = poses[..., 3].mean(0)  # (3)
    
    # 2. Compute the z axis
    #z = normalize(poses[..., 2].mean(0))  # (3)
    # 3. Compute axis y' (no need to normalize as it's not the final output)
    #y_ = poses[..., 1].mean(0)  # (3)
    # 4. Compute the x axis
    #x = normalize(np.cross(z, y_))  # (3)
    #x = normalize(np.cross(y_, z))  # (3)
    
    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    #y = np.cross(x, z)  # (3)
    #y = np.cross(z, x)
    #pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)
    
    center = poses[..., 3].mean(0)  # (3)
    z = normalize(poses[..., 2].mean(0))  # (3)
    y_ = poses[..., 0].sum(0)  # (3)
    y = normalize(np.cross(z, y_))
    x = normalize(np.cross(y, z))  # (3)
    
    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)
    
    return pose_avg


def center_poses(poses, blender2opencv):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """
    poses = poses @ blender2opencv
    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[
        :3
    ] = pose_avg  # convert to homogeneous coordinate for faster computation
    pose_avg_homo = pose_avg_homo
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = np.concatenate(
        [poses, last_row], 1
    )  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    #     poses_centered = poses_centered  @ blender2opencv
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, pose_avg_homo


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec0_avg = up
    vec1 = normalize(np.cross(vec2, vec0_avg))
    vec0 = normalize(np.cross(vec1, vec2))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def render_path_axis_param(c2w, up, ax, rad, focal, view_range, N):
    c2w = c2w[:3,:4]
    render_poses = []
    center = c2w[:,3]
    #hwf = c2w[:,4:5]
    v = c2w[:,ax] * rad
    for t in np.linspace(-view_range,view_range,N+1)[:-1]:
        c = center + t * v
        #z = normalize(c - (c - focal * c2w[:,2]))
        z = normalize(c - (center - focal * c2w[:,2]))
        render_poses.append(viewmatrix(z, up, c))
    return render_poses

def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts.T-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt

def get_axis(c2ws_all, near_fars, axis, focal, view_range, N_views=120):
    """
    Generate a set of poses using NeRF's spiral camera trajectory as validation poses.
    """
    # center pose
    #c2w = average_poses(c2ws_all)
    arr = c2ws_all[:,0,3]
    id = np.argpartition(arr, len(arr) // 2)[len(arr) // 2]
    c2w = c2ws_all[id,:,:]
    # Get average pose
    up = normalize(c2ws_all[:, :3, 0].sum(0))

    # Find a reasonable "focus depth" for this dataset
    dt = 0.9
    close_depth, inf_depth = near_fars.min() * 0.9, near_fars.max() * 5.0
    #focal = 1.0 / ((1.0 - dt) / close_depth + dt / inf_depth)
    # Get radii for spiral path
    shrink_factor = .8
    zdelta = close_depth * .2
    tt = ptstocam(c2ws_all[:, :3,3].T, c2w).T
    rads = np.percentile(np.abs(tt), 90, -1)
    #print("get_axis", focal, view_range)
    render_poses = render_path_axis_param(c2w, up, axis, shrink_factor*rads[axis], focal, view_range, N=N_views)
    return np.stack(render_poses)

def get_video_cam_infos_x_axis(cam_extrinsics, cam_intrinsics, datadir, focal, view_range, N_views):
    height=cam_intrinsics[1].height
    width=cam_intrinsics[1].width
    FovY = focal2fov(cam_intrinsics[1].params[0], height)
    FovX = focal2fov(cam_intrinsics[1].params[0], width)
    
    poses_arr = np.load(os.path.join(datadir, "poses_bounds_scview.npy"))
    
    poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)
    near_fars = poses_arr[:, -2:]
    
    val_poses = get_axis(poses, near_fars, 1, focal, view_range, N_views=N_views)
    val_poses = np.concatenate([val_poses[...,1:2], -val_poses[...,0:1], val_poses[...,2:]], -1)
    views = []
    trans = np.array([0.0, 0.0, 0.0])
    scale = 1.0
    zfar = 100.0
    znear = 0.01

    for idx, p in enumerate(val_poses):
        pose = np.eye(4)
        pose[:3,:] = p[:3,:]
        R = pose[:3,:3]
        
        R = - R
        R[:,0] = -R[:,0]
        T = -pose[:3,3].dot(R)
        world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
        projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=FovX, fovY=FovY).transpose(0,1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]
        views.append(ViewInfo(uid=idx, height=height, width=width, FovY=FovY, FovX=FovX, world_view_transform=world_view_transform, full_proj_transform=full_proj_transform, camera_center=camera_center))
    return views