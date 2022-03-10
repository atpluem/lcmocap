from typing import Optional, Tuple

import os
import os.path as osp

import numpy as np

from torch.utils.data import Dataset
from loguru import logger

class PKLFolder(Dataset):
    def __init__(
        self,
        pkl_folder: str,
        transforms=None,
        exts: Optional[Tuple] = None
    ) -> None:

        if exts is None:
            exts = ['.pkl']

        self.pkl_folder = osp.expandvars(pkl_folder)

        logger.info(f'Building pkl folder dataset for folder: {self.pkl_folder}')

        self.data_paths = np.array([
            osp.join(self.pkl_folder, fname)
            for fname in os.listdir(self.pkl_folder)
            if any(fname.endswith(ext) for ext in exts)
        ])
        self.num_items = len(self.data_paths)

    def __len__(self) -> int:
        return self.num_items

    def __getitem__(self, index):
        pkl_path = self.data_paths[index]

        return {'pkl_path': pkl_path}
        