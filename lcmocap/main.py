import os
import os.path as osp
import sys
import torch
from torch.utils import data
import openpose as op
import smplifyx as smx

from config import parse_args
from data import build_dataloader
from tqdm import tqdm
from loguru import logger
from smplx import build_layer

def main() -> None:
    
    # ===== Openpose API  =====
    # keypoints = op.openposeAPI()

    # Read YAML config
    config = parse_args()

    # Defind CUDA device
    device = torch.device('cuda')
    if not torch.cuda.is_available():
        logger.error('CUDA is not available!')
        sys.exit(3)

    # Initialize tqdm
    logger.remove()
    logger.add(lambda x: tqdm.write(x, end=''), level=config.logger_level.upper(),
                colorize=True)

    # Defind output folder
    output_folder = osp.expanduser(osp.expandvars(config.output_folder))
    logger.info(f'Saving output to: {output_folder}')
    os.makedirs(output_folder, exist_ok=True)

    # Build layer of template model (SMPL, SMPLX, ...)
    # model_path = config.body_model.folder
    # body_model = build_layer(model_path, **config.body_model)
    # logger.info(body_model)
    # body_model = body_model.to(device=device)

    # Dataloader
    data_obj_dict = build_dataloader(config)
    print(data_obj_dict['sourceloader'])

if __name__ == "__main__":
    main()