from functools import partial
import os
import os.path as osp
import sys
from numpy.lib.utils import source
import torch
import openpose as op
import smplifyx.smplifyx as smx

from tqdm import tqdm
from loguru import logger
from smplx import build_layer
from config import parse_args
from data import build_dataloader
from retarget import run_retarget
from smplifyx.camera import create_camera

def main() -> None:
    
    # ===== Openpose API  =====
    # keypoints = op.openposeAPI()

    # ===== Retargeting =====
    # Read YAML config 
    config = parse_args()

    # Initialize tqdm
    logger.remove()
    logger.add(lambda x: tqdm.write(x, end=''), level=config.logger_level.upper(),
                colorize=True)

    # Defind output folder
    output_folder = osp.expanduser(osp.expandvars(config.output_folder))
    logger.info(f'Saving output to: {output_folder}')
    os.makedirs(output_folder, exist_ok=True)

    # Build layer of template model (SMPL, SMPLX, ...)
    model_path = config.body_model.folder
    body_model = build_layer(model_path, **config.body_model)
    logger.info(body_model)

    # Dataloader
    data_obj_dict = build_dataloader(config)
    source_mesh = data_obj_dict['source_pose']
    source_std_mesh = data_obj_dict['source_std_pose']
    target_mesh = data_obj_dict['target_pose']

    # Fitting Retarget
    run_retarget(config=config,
                source_mesh=source_mesh,
                source_std_mesh=source_std_mesh,
                target_mesh=target_mesh,
                visualize=False)

if __name__ == "__main__":
    main()