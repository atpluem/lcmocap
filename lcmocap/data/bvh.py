import os
import os.path as osp
import numpy as np
import logging

from torch.utils.data import Dataset
from loguru import logger
from typing import Optional, Tuple
from data.bvh_parser import BVH_file

class BVHFolder(Dataset):
    def __init__(
        self,
        bvh_folder: str,
        transforms=None,
        exts: Optional[Tuple] = None
    ) -> None:

        if exts is None:
            exts = ['.bvh']

        self.bvh_folder = osp.expandvars(bvh_folder)

        logger.info(f'Building bvh folder dataset for folder: {self.bvh_folder}')

        self.data_paths = np.array([
            osp.join(self.bvh_folder, fname)
            for fname in os.listdir(self.bvh_folder)
            if any(fname.endswith(ext) for ext in exts)
        ])
        self.num_items = len(self.data_paths)

    def __len__(self) -> int:
        return self.num_items

    def __getitem__(self, index):
        bvh_path = self.data_paths[index]

        # Load the BVH
        file = BVH_file(bvh_path)

        return {'edges': np.asarray(file.edges), 
                'names': np.asarray(file.names)}