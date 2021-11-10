from os import name
from omegaconf import OmegaConf
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class MeshFolder:
    mesh_folder: str = ''

@dataclass
class DatasetConfig:
    num_workers: int = 0
    name: str = 'mesh-folder'
    source_folder: MeshFolder = MeshFolder()
    target_folder: MeshFolder = MeshFolder()

@dataclass
class BodyModel:
    name: str = 'body-model'
    model_type: str = 'smpl'
    gender: str = 'neutral'
    ext: str = 'pkl'
    folder: str = '../models'

@dataclass
class Config:
    use_cuda: bool = False
    log_file: str = '/tmp/logs'
    output_folder: str = 'output/retargeting'
    save_verts: bool = True
    save_joints: bool = True
    save_mesh: bool = False
    logger_level: str = 'INFO'
    batch_size: Optional[int] = 1

    deformation_transfer_path: str = ''
    mask_ids_fname: str = ''

    datasets: DatasetConfig = DatasetConfig()
    body_model: BodyModel = BodyModel()

conf = OmegaConf.structured(Config)  