import sys
import numpy as np
import torch
from torch._C import dtype
import torch.nn as nn

from tqdm import tqdm
from loguru import logger
from typing import Optional, Dict, Callable
from utils import (Tensor)

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
    batch: Dict[str, Tensor],
    body_model: nn.Module
) -> Dict[str, Tensor]:
    '''
        Runs fitting
    '''
    vertices = batch['vertices']
    faces = batch['faces']

    batch_size = len(vertices)
    dtype, device = vertices.dtype, vertices.device

    # Get the parameters from the model
    var_dict = get_variables(batch_size, body_model)

    # Build the optimizer object for the current batch
    optim = config.get('optim', {})

    