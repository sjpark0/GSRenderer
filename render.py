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
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
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
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    args = get_combined_args(parser)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    model_path = "data"
    out_path = "data/output"
    frame_num = 50
    total_frame = 100
    num_views = 49
    focal = 80.0
    view_range = 1.0

    
    makedirs(os.path.join(out_path, str(frame_num)), exist_ok=True)
    
    renderer = SJRendererLKG.SJRendererLKG(model_path, hyperparam.extract(args), out_path, focal, view_range, num_views, total_frame)
    time1 = time()
    render_list = renderer.rendering(50)    
    time2 = time()
    print("FPS:",(num_views/(time2-time1)))
        
    multithread_write(render_list, os.path.join(out_path, str(frame_num)))
