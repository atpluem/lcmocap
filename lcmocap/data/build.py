import sys
from traceback import print_tb
from numpy.lib.utils import source
import torch
import torch.utils.data as dutils

from typing import List, Tuple
from loguru import logger
from torch.utils.data import dataset
from .mesh import MeshFolder
from .fbx import FBXFolder
from .pose_pkl import PKLFolder

def build_dataloader(config):
    dset_name = config.datasets.name

    if dset_name == 'data-folder':
        ''' Load Meshs '''
        # source_folder = config.datasets.src_mesh_dir
        # source_std_folder = config.datasets.src_std_mesh_dir
        # target_folder = config.datasets.dest_mesh_dir
        # logger.info(source_folder.pretty())
        # logger.info(source_std_folder.pretty())
        # logger.info(target_folder.pretty())
        # source_data = MeshFolder(**source_folder)
        # source_std_data = MeshFolder(**source_std_folder)
        # target_data = MeshFolder(**target_folder)
        ''' Load BVH '''
        # source_folder = config.datasets.src_bvh_dir
        # target_folder = config.datasets.dest_bvh_dir
        # logger.info(source_folder.pretty())
        # logger.info(target_folder.pretty())
        # source_bvh = BVHFolder(**source_folder)
        ''' Load Mesh and FBX '''
        source_folder = config.datasets.src_dir
        source_std_folder = config.datasets.src_std_dir
        target_folder = config.datasets.dest_dir
        pose_folder = config.datasets.pose_params_dir
        logger.info(source_folder)
        logger.info(source_std_folder)
        logger.info(target_folder)
        logger.info(pose_folder)
        source_mesh = MeshFolder(source_folder)
        source_std_mesh = MeshFolder(source_std_folder)
        target_mesh = MeshFolder(target_folder)
        source_fbx = FBXFolder(source_folder)
        target_fbx = FBXFolder(target_folder)
        pose_pkl = PKLFolder(pose_folder)

    else:
        raise ValueError(f'Unknown dataset: {dset_name}')

    # batch_size = config.batch_size
    # num_workers = config.datasets.num_workers
    
    # logger.info(f'Creating dataloader with B={batch_size}, workers={num_workers}')

    # sourceloader = dutils.DataLoader(source_data, batch_size=batch_size,
    #                                 num_workers=num_workers, shuffle=False)
    # targetloader = dutils.DataLoader(target_data, batch_size=batch_size,
    #                                 num_workers=num_workers, shuffle=False)
    
    # return {'sourceloader': sourceloader, 'source_data': source_data,
    #         'targetloader': targetloader, 'target_data': target_data}

    return {'src_fbx_path': source_fbx[0], 'dest_fbx_path': target_fbx[0],
            'pose_params_path': pose_pkl[0], 'src_mesh_std': source_std_mesh[0],
            'src_mesh': source_mesh[0], 'dest_mesh': target_mesh[0]}

    