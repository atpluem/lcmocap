import sys
import numpy as np
from pyrender import light
import torch
import torch.nn as nn
import fitting

from tqdm import tqdm
from loguru import logger
from typing import Optional, Dict, Callable
from data import mesh
from utils import (Tensor)
from human_body_prior.tools.model_loader import load_vposer
from mesh_viewer import MeshViewer as mv

def get_variables(
    batch_size: int,
    body_model: nn.Module,
    dtype: torch.dtype = torch.float32
) -> Dict[str, Tensor]:
    var_dict = {}

    device = next(body_model.buffers()).device

    if (body_model.name() == 'SMPL' or body_model.name() == 'SMPL+H' or
            body_model.name() == 'SMPL-X'):
        var_dict.update({
            'transl': torch.zeros(
                [batch_size, 3], device=device, dtype=dtype),
            'global_orient': torch.zeros(
                [batch_size, 1, 3], device=device, dtype=dtype),
            'body_pose': torch.zeros(
                [batch_size, body_model.NUM_BODY_JOINTS, 3],
                device=device, dtype=dtype),
            'betas': torch.zeros([batch_size, body_model.num_betas],
                                 dtype=dtype, device=device),
        })

    if body_model.name() == 'SMPL+H' or body_model.name() == 'SMPL-X':
        var_dict.update(
            left_hand_pose=torch.zeros(
                [batch_size, body_model.NUM_HAND_JOINTS, 3], device=device,
                dtype=dtype),
            right_hand_pose=torch.zeros(
                [batch_size, body_model.NUM_HAND_JOINTS, 3], device=device,
                dtype=dtype),
        )

    if body_model.name() == 'SMPL-X':
        var_dict.update(
            jaw_pose=torch.zeros([batch_size, 1, 3],
                                 device=device, dtype=dtype),
            leye_pose=torch.zeros([batch_size, 1, 3],
                                  device=device, dtype=dtype),
            reye_pose=torch.zeros([batch_size, 1, 3],
                                  device=device, dtype=dtype),
            expression=torch.zeros(
                [batch_size, body_model.num_expression_coeffs],
                device=device, dtype=dtype),
        )

    # Toggle gradients to True
    for key, val in var_dict.items():
        val.requires_grad_(True)

    return var_dict

def run_fitting(
    config,
    source_mesh,
    target_mesh,
    camera,
    body_model: nn.Module,
    use_cuda=True,
    batch_size=1,
    dtype=torch.float32,
    visualize=False
) -> Dict[str, Tensor]:
    '''
        Runs fitting
    '''
    assert batch_size == 1, 'PyTorch L-BFGS only supports batch_size == 1'

    device = torch.device('cuda') if use_cuda else torch.device('cpu')

    # Get the parameters from the model
    var_dict = get_variables(batch_size, body_model)

    # Build the optimizer object for the current batch
    optim = config.get('optim', {})

    # Preprocess
    

    with fitting.FittingMonitor() as monitor:
        monitor.run_fitting()
    
    if visualize:
        import trimesh
        
        mesh = trimesh.Trimesh(source_mesh['vertices'], 
                               source_mesh['faces'], process=False)

        mesh.show()