"""
This code comes from https://github.com/rubenvillegas/cvpr2018nkn/blob/master/datasets/fbx2bvh.py
"""
import os
import os.path as osp
import numpy as np
import bpy

from torch.utils.data import Dataset
from typing import Optional, Tuple
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

        logger.info(f'Building mesh folder dataset for folder: {self.fbx_folder}')

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
        dump_path = self.data_paths[index] + '.bvh'

        # Load the FBX
        bpy.ops.import_scene.fbx(filepath=fbx_path)

        frame_start = 9999
        frame_end = -9999
        action = bpy.data.actions[-1]
        if action.frame_range[1] > frame_end:
            frame_end = action.frame_range[1]
        if action.frame_range[0] < frame_start:
            frame_start = action.frame_range[0]

        frame_end = np.max([60, frame_end])
        bpy.ops.export_anim.bvh(filepath=dump_path,
                                frame_start=frame_start,
                                frame_end=frame_end,
                                root_transform_only=True)
        bpy.data.actions.remove(bpy.data.action[-1])

        # return {
        #     'vertices': np.asarray(mesh.vertices, dtype=np.float32),
        #     'faces': np.asarray(mesh.faces, dtype=np.int32),
        #     'indices': index,
        #     'paths': mesh_path,
        # }