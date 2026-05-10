import imageio
import numpy as np
import torch
import os
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from time import time
import concurrent.futures
from scene.scene_info import readSCViewinfo
from scene import Scene

def multithread_write(image_list, path):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)
    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(image, os.path.join(path, '{0:05d}'.format(count) + ".png"))
            return count, True
        except:
            return count, False
        
    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index, path)
    
def render_set(out_path, views, gaussians, background):    
    makedirs(out_path, exist_ok=True)
    render_list = []
    print("point nums:",gaussians._xyz.shape[0])
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if idx == 0:time1 = time()
        
        rendering = render(view, gaussians, background)
        render_list.append(rendering)
    
    time2=time()
    print("FPS:",(len(views)-1)/(time2-time1))    
    multithread_write(render_list, out_path)

    
def render_sets(model_path, hyperparam, out_path, focal : float, view_range : float):
    with torch.no_grad():
        gaussians = GaussianModel(3, hyperparam)
        scene_info = readSCViewinfo(model_path, focal, view_range)
        scene = Scene(scene_info, gaussians, focal=focal, view_range=view_range)
        
        bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        #for idx, video_camera in enumerate(scene.getVideoCameras()):
        #    render_set(out_path,video_camera,gaussians,pipeline,background,cam_type)
        render_set(out_path,scene.video_camera[0],gaussians,background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    hyperparam = ModelHiddenParams(parser)
    args = get_combined_args(parser)
    #print("Rendering " , args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    render_sets("data", hyperparam.extract(args), "data/output", 80, 1.0)


#parser = ArgumentParser(description="Testing script parameters")
#hyperparam = ModelHiddenParams(parser)
#render_sets("data", "data/output", 1, hyperparam, 80.0, 1.0)