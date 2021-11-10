import sys
from numpy.lib.utils import source
import torch
import torch.utils.data as dutils

from typing import List, Tuple
from loguru import logger
from torch.utils.data import dataset
from .mesh import MeshFolder

def build_dataloader(config):
    dset_name = config.datasets.name

    if dset_name == 'mesh-folder':
        source_folder = config.datasets.source_folder
        target_folder = config.datasets.target_folder
        logger.info(source_folder.pretty())
        logger.info(target_folder.pretty())
        source_data = MeshFolder(**source_folder)
        target_data = MeshFolder(**target_folder)
    else:
        raise ValueError(f'Unknown dataset: {dset_name}')

    batch_size = config.batch_size
    num_workers = config.datasets.num_workers
    
    logger.info(f'Creating dataloader with B={batch_size}, workers={num_workers}')

    sourceloader = dutils.DataLoader(source_data, batch_size=batch_size,
                                    num_workers=num_workers, shuffle=False)
    targetloader = dutils.DataLoader(target_data, batch_size=batch_size,
                                    num_workers=num_workers, shuffle=False)
    
    return {'sourceloader': sourceloader, 'source_data': source_data,
            'targetloader': targetloader, 'target_data': target_data}