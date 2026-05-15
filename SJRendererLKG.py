
import math
import torch
import os
import numpy as np

from tqdm import tqdm
from scene.gaussian_model import GaussianModel
from scene.pose import get_video_cam_infos_x_axis
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
class SJRendererLKG:
    def __init__(self, model_path, hyperparam, out_path, focal, view_range, num_views, total_frame):
        self.model_path = model_path
        self.hyperparam = hyperparam
        self.out_path = out_path
        self.focal = focal
        self.view_range = view_range
        self.total_frame = total_frame
        self.num_views = num_views
        with torch.no_grad():
            self.gaussians = GaussianModel(3, self.hyperparam)
            self.gaussians.load_gaussian_model(self.model_path)
            self.video_camera = self.readViewParam()
            bg_color = [0, 0, 0]
            self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            
    def render(self, viewpoint_camera, frame = 0.0):
        """
        Render the scene. 
        
        Background tensor (bg_color) must be on GPU!
        """        
        means3D = self.gaussians.get_xyz
        tanfovx = math.tan(viewpoint_camera.FovX * 0.5)
        tanfovy = math.tan(viewpoint_camera.FovY * 0.5)
        
        raster_settings = GaussianRasterizationSettings(image_height=viewpoint_camera.height, image_width=viewpoint_camera.width, tanfovx=tanfovx, tanfovy=tanfovy, bg=self.background, scale_modifier=1.0, viewmatrix=viewpoint_camera.world_view_transform.cuda(), projmatrix=viewpoint_camera.full_proj_transform.cuda(), sh_degree=self.gaussians.active_sh_degree, campos=viewpoint_camera.camera_center.cuda(), prefiltered=False, debug=False)
        
        time = torch.tensor(frame).to(means3D.device).repeat(means3D.shape[0],1)
        

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        
        opacity = self.gaussians._opacity
        shs = self.gaussians.get_features
        scales = self.gaussians._scaling
        rotations = self.gaussians._rotation
        
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = self.gaussians._deformation(means3D, scales, rotations, opacity, shs, time)
        
        scales_final = self.gaussians.scaling_activation(scales_final)
        rotations_final = self.gaussians.rotation_activation(rotations_final)
        opacity = self.gaussians.opacity_activation(opacity_final)
        
        rendered_image, _, _ = rasterizer(means3D = means3D_final, means2D = None, shs = shs_final, colors_precomp = None, opacities = opacity, scales = scales_final, rotations = rotations_final, cov3D_precomp = None)
        return rendered_image

    def readViewParam(self):
        cameras_extrinsic_file = os.path.join(self.model_path, "sparse_/images.bin")
        cameras_intrinsic_file = os.path.join(self.model_path, "sparse_/cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)        
        video_cam_info = get_video_cam_infos_x_axis(cam_extrinsics, cam_intrinsics, self.model_path, self.focal, self.view_range, self.num_views)

        return video_cam_info
    
    def rendering(self, frame_num):
        render_list = []
        with torch.no_grad():
            for idx, view in enumerate(tqdm(self.video_camera, desc="Rendering progress")):
                rendering = self.render(view, frame_num / self.total_frame)
                render_list.append(rendering)
            
        return render_list