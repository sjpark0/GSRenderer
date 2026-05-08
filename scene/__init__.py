#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json

from scene.gaussian_model import GaussianModel
from scene.dataset import FourDGSdataset
from torch.utils.data import Dataset

class Scene:

    gaussians : GaussianModel

    def __init__(self, scene_info, gaussians : GaussianModel, focal=10.0, view_range=1.0):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.gaussians = gaussians
        self.maxtime = scene_info.maxtime
        self.model_path = scene_info.model_path
        
        self.video_camera = []
        for video_camera in scene_info.video_cameras:
            self.video_camera.append(FourDGSdataset(video_camera))
        
        xyz_max = scene_info.point_cloud.points.max(axis=0)
        xyz_min = scene_info.point_cloud.points.min(axis=0)
        
        self.gaussians._deformation.deformation_net.set_aabb(xyz_max,xyz_min)
        self.gaussians.load_ply(os.path.join(self.model_path, "point_cloud.ply"))
        self.gaussians.load_model(self.model_path)
        
    def getVideoCameras(self, scale=1.0):
        return self.video_camera
