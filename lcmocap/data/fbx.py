from typing import Optional, Tuple

import os
import os.path as osp

import numpy as np
import bpy

from torch.utils.data import Dataset
from loguru import logger

class FBXFolder(Dataset):
    def __init__(
        self,
        fbx_folder: str,
        transforms=None,
        exts: Optional[Tuple] = None
    ) -> None:

        if exts is None:
            exts = ['.fbx']

        self.fbx_folder = osp.expandvars(fbx_folder)

        logger.info(f'Building fbx folder dataset for folder: {self.fbx_folder}')

        self.data_paths = np.array([
            osp.join(self.fbx_folder, fname)
            for fname in os.listdir(self.fbx_folder)
            if any(fname.endswith(ext) for ext in exts)
        ])
        self.num_items = len(self.data_paths)

    def __len__(self) -> int:
        return self.num_items

    def __getitem__(self, index):
        fbx_path = self.data_paths[index]

        return {'fbx_path': fbx_path}
        