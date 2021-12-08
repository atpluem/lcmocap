import sys
import os

import argparse

from omegaconf import OmegaConf
from .defaults import conf as default_conf

def parse_args(argv=None) -> OmegaConf:
    arg_formatter = argparse.ArgumentDefaultsHelpFormatter

    description = 'Retargeting script'
    parser = argparse.ArgumentParser(formatter_class=arg_formatter,
                                    description=description)

    parser.add_argument('--config', type=str, dest='config',
                        help='The configuration of retargeting')

    cmd_args = parser.parse_args()

    cfg = default_conf.copy()
    
    # Load parameter from YAML then merge with default config
    if cmd_args.config:
        cfg.merge_with(OmegaConf.load(cmd_args.config))
    
    return cfg
