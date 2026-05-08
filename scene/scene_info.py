import os
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from scene.gaussian_model import BasicPointCloud
import numpy as np
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from tqdm import tqdm
from plyfile import PlyData, PlyElement
from scene.pose import get_video_cam_infos_x_axis


class SceneInfo(NamedTuple):
    model_path: str
    point_cloud: BasicPointCloud
    video_cameras: list
    ply_path: str
    maxtime: int


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def readSCViewinfo(datadir,focal, view_range, llffhold=8):

    cameras_extrinsic_file = os.path.join(datadir, "sparse_/images.bin")
    cameras_intrinsic_file = os.path.join(datadir, "sparse_/cameras.bin")
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    
    image_length = 100
    video_cam_infos = []
    for frame_idx in range(image_length):
        time = float(frame_idx / image_length)
        video_cam_infos.append(get_video_cam_infos_x_axis(cam_extrinsics, cam_intrinsics, datadir, focal, view_range, time))
    ply_path = os.path.join(datadir, "points3D_scview.ply")
    
    try:
        pcd = fetchPly(ply_path)
        
    except:
        pcd = None
    
    scene_info = SceneInfo(model_path=datadir, point_cloud=pcd, video_cameras=video_cam_infos, maxtime=0, ply_path=ply_path)
    return scene_info