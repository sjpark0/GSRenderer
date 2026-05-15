import torch
from scene.deformation import deform_network
from scene.deformation import DeformationParam
import os
path = "data"
param = DeformationParam(net_width=128, timebase_pe=4, defor_depth=1, posebase_pe=10, scale_rotation_pe=2, opacity_pe=2, 
                             timenet_width=64, timenet_output=32, grid_pe=0, no_grid=False, bounds=1.6, 
                             kplanes_config={'grid_dimensions': 2, 'input_coordinate_dim': 4, 'output_coordinate_dim': 16, 'resolution': [64, 64, 64, 150]},
                             multires=[1,2], empty_voxel=False, static_mlp=False, no_dx=False, no_ds=False, no_dr=False, no_do=False, no_dshs=False, apply_rotation=False)
deformation = deform_network(param)
weight_dict = torch.load(os.path.join(path,"deformation.pth"),map_location="cuda")
deformation.load_state_dict(weight_dict)
deformation.eval()

example_input = (torch.rand(135376, 3), torch.rand(135376, 3), torch.rand(135376, 4), torch.rand(135376, 1), torch.rand(135376, 16, 3), torch.rand(135376, 1))
traced_model = torch.jit.trace(deformation, example_input)
traced_model.save("traced_model.pt")