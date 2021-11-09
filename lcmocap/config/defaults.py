from os import name
from omegaconf import OmegaConf
from dataclasses import dataclass

@dataclass
class MeshFolder:
    source_folder: str = '../input/source_model'
    target_folder: str = '../input/target_model'

@dataclass
class DatasetConfig:
    name: str = 'mesh-folder'
    mesh_folder: MeshFolder = MeshFolder()

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

    datasets: DatasetConfig = DatasetConfig()
    body_model: BodyModel = BodyModel()

conf = OmegaConf.structured(Config)  