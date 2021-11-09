from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os

import os.path as osp

import time
import yaml
import torch

import smplx

def smplifyx(**args):
    try:
        print(__name__)

    except Exception as e:
        print(e)
        sys.exit(-1)

if __name__ == "__main__":
    try:
        # Set path to smplifyx that contains python modules 
        sys.path.append('/home/pluem/LCMocap/smplify-x/smplifyx')
        
        # Import SMPLify-x library
        from utils import JointMapper
        from cmd_parser import parse_config
        from data_parser import create_dataset
        from fit_single_frame import fit_single_frame
        from camera import create_camera
        from prior import create_prior
        torch.backends.cudnn.enabled = False
    except ImportError as e:
        print('Error: SMPLify-x library could not be found.')
        raise e

    args = parse_config()
    smplifyx(**args)