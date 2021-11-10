from typing import Optional, Tuple

import os
import os.path as osp

import numpy as np
from psbody.mesh import Mesh
import trimesh

from torch.utils.data import Dataset
from loguru import logger


class MeshFolder(Dataset):
    def __init__(
        self,
        mesh_folder: str,
        transforms=None,
        exts: Optional[Tuple] = None
    ) -> None:
        ''' Dataset similar to ImageFolder that reads meshes with the same
            topology
        '''
        if exts is None:
            exts = ['.obj', '.ply']

        self.mesh_folder = osp.expandvars(mesh_folder)

        logger.info(f'Building mesh folder dataset for folder: {self.mesh_folder}')

        self.data_paths = np.array([
            osp.join(self.mesh_folder, fname)
            for fname in os.listdir(self.mesh_folder)
            if any(fname.endswith(ext) for ext in exts)
        ])
        self.num_items = len(self.data_paths)

    def __len__(self) -> int:
        return self.num_items

    def __getitem__(self, index):
        mesh_path = self.data_paths[index]

        # Load the mesh
        mesh = trimesh.load(mesh_path, process=False)

        return {
            'vertices': np.asarray(mesh.vertices, dtype=np.float32),
            'faces': np.asarray(mesh.faces, dtype=np.int32),
            'indices': index,
            'paths': mesh_path,
        }