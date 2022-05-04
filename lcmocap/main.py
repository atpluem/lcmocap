import os
import os.path as osp
import sys
import torch
import openpose as op

from tqdm import tqdm
from loguru import logger
# from smplx import build_layer
from config import parse_args
from data import build_dataloader
from retarget_mesh import run_retarget_mesh
from retarget import run_retarget

def main() -> None:
    # Read YAML config 
    config = parse_args()

    # Initialize tqdm
    logger.remove()
    logger.add(lambda x: tqdm.write(x, end=''), level=config.logger_level.upper(), colorize=True)
    
    ##############################################################
    ##                     Openpose API                         ##
    ##############################################################

    # keypoints = op.openposeAPI()

    ##############################################################
    ##                     Retargeting                          ##
    ##############################################################
    
    # Defind output folder
    output_folder = osp.expanduser(osp.expandvars(config.output_folder))
    logger.info(f'Define output to: {output_folder}')
    os.makedirs(output_folder, exist_ok=True)

    # Build layer of template model (SMPL, SMPLX, ...)
    # model_path = config.body_model.folder
    # body_model = build_layer(model_path, **config.body_model)
    # logger.info(body_model)

    # Dataloader
    data_obj_dict = build_dataloader(config)
    ''' Load Meshs '''
    # src_mesh = data_obj_dict['src_mesh']
    # src_std_mesh = data_obj_dict['src_mesh_std']
    # dest_mesh = data_obj_dict['dest_mesh']
    ''' Load BVH '''
    # source_bvh = data_obj_dict['']
    ''' Load FBX '''
    src_fbx_path = data_obj_dict['src_fbx_path']
    dest_fbx_path = data_obj_dict['dest_fbx_path']
    pose_params_path = data_obj_dict['pose_params_path']

    # Fitting Retarget MESH
    # run_retarget_mesh(config=config,
    #                   src_mesh=src_mesh,
    #                   src_std_mesh=src_std_mesh,
    #                   dest_mesh=dest_mesh,
    #                   out_path=output_folder,
    #                   visualize=False)

    # Fitting Retarget RIGGING
    run_retarget(config=config,
                pose_params_path=pose_params_path['pkl_path'],
                src_path=src_fbx_path['fbx_path'],
                dest_path=dest_fbx_path['fbx_path'],
                out_path=output_folder,
                visualize=False)

if __name__ == "__main__":
    main()