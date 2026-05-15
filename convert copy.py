import torch
from scene.deformation import deform_network
import os
from utils.params_utils import merge_hparams
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from utils.general_utils import safe_state
from argparse import ArgumentParser

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


path = "data"
deformation = deform_network(hyperparam.extract(args))
weight_dict = torch.load(os.path.join(path,"deformation.pth"),map_location="cuda")
deformation.load_state_dict(weight_dict)
deformation.eval()

example_input = (torch.rand(135376, 3), torch.rand(135376, 3), torch.rand(135376, 4), torch.rand(135376, 1), torch.rand(135376, 16, 3), torch.rand(135376, 1))
traced_model = torch.jit.trace(deformation, example_input)
traced_model.save("traced_model.pt")