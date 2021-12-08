from os import name
from omegaconf import OmegaConf
from dataclasses import dataclass
from typing import Tuple, Optional

from .body_model_defaults import conf as body_model_cfg,  BodyModelConfig
from .optim_defaults import conf as optim_cfg, OptimConfig

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
class BodyPartSegm:
    name: str = 'body-part segmentation path'
    smpl_path: str = ''

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
    
    optim: OptimConfig = optim_cfg
    datasets: DatasetConfig = DatasetConfig()
    body_model: BodyModelConfig = body_model_cfg
    body_part_segm: BodyPartSegm = BodyPartSegm()

    deformation_transfer_path: str = ''
    mask_ids_fname: str = ''

conf = OmegaConf.structured(Config)  