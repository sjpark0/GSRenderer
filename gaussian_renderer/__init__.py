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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from time import time as get_time
def render(viewpoint_camera, pc : GaussianModel, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, cam_type=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
     
    means3D = pc.get_xyz
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    raster_settings = GaussianRasterizationSettings(image_height=int(viewpoint_camera.image_height), image_width=int(viewpoint_camera.image_width), tanfovx=tanfovx, tanfovy=tanfovy, bg=bg_color, scale_modifier=scaling_modifier, viewmatrix=viewpoint_camera.world_view_transform.cuda(), projmatrix=viewpoint_camera.full_proj_transform.cuda(), sh_degree=pc.active_sh_degree, campos=viewpoint_camera.camera_center.cuda(), prefiltered=False, debug=False)
    
    time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
    

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    opacity = pc._opacity
    shs = pc.get_features
    scales = pc._scaling
    rotations = pc._rotation
    
    means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(means3D, scales, rotations, opacity, shs, time)
    
    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)
    
    rendered_image, _, _ = rasterizer(means3D = means3D_final, means2D = None, shs = shs_final, colors_precomp = None, opacities = opacity, scales = scales_final, rotations = rotations_final, cov3D_precomp = None)
    
    return rendered_image

