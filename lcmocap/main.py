from logging import log
import os
import os.path as osp
import sys
import openpose as op
import smplifyx as smx

from config import parse_args
from tqdm import tqdm
from loguru import logger

def main() -> None:
    
    # ===== Openpose API  =====
    # keypoints = op.openposeAPI()

    # Read YAML config
    config = parse_args()
    
    logger.remove()
    logger.add(lambda x: tqdm.write(x, end=''), level=config.logger_level.upper(),
                colorize=True)

    # Defind output folder
    output_folder = osp.expanduser(osp.expandvars(config.output_folder))
    logger.info(f'Saving output to: {output_folder}')
    os.makedirs(output_folder, exist_ok=True)



if __name__ == "__main__":
    main()