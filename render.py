import imageio
import numpy as np
import SJRendererLKG
import os
import torchvision
import concurrent.futures

from utils.general_utils import safe_state
from argparse import ArgumentParser
from os import makedirs
from time import time
from utils.params_utils import merge_hparams
from scene.deformation import DeformationParam
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

def merge_hparams1(args, config):    
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)

    return args

if __name__ == "__main__":
    # Set up command line argument parser    
    param = DeformationParam(net_width=128, timebase_pe=4, defor_depth=1, posebase_pe=10, scale_rotation_pe=2, opacity_pe=2, 
                             timenet_width=64, timenet_output=32, grid_pe=0, no_grid=False, bounds=1.6, 
                             kplanes_config={'grid_dimensions': 2, 'input_coordinate_dim': 4, 'output_coordinate_dim': 16, 'resolution': [64, 64, 64, 150]},
                             multires=[1,2], empty_voxel=False, static_mlp=False, no_dx=False, no_ds=False, no_dr=False, no_do=False, no_dshs=False, apply_rotation=False)
    
    model_path = "data"
    out_path = "data/output"
    frame_num = 50
    total_frame = 100
    num_views = 49
    focal = 80.0
    view_range = 1.0

    
    makedirs(os.path.join(out_path, str(frame_num)), exist_ok=True)
    
    renderer = SJRendererLKG.SJRendererLKG(model_path, param, out_path, focal, view_range, num_views, total_frame)
    time1 = time()
    render_list = renderer.rendering(50)    
    time2 = time()
    print("FPS:",(num_views/(time2-time1)))
        
    multithread_write(render_list, os.path.join(out_path, str(frame_num)))
