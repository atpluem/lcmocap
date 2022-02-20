from tkinter.tix import Tree
from traceback import print_tb
from unittest import result
from matplotlib import markers, projections
from matplotlib.pyplot import axis
from mpl_toolkits.mplot3d import Axes3D
from networkx.algorithms.cluster import triangles
import numpy as np
import torch
import torch.nn as nn
import trimesh
import networkx as nx
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import math
import seaborn as sns
import bpy

from tkinter import *
from tqdm import tqdm
from loguru import logger
from typing import Optional, Dict, Callable, Union
from data import mesh
from utils import (Tensor)
from human_body_prior.tools.model_loader import load_vposer
from shape import *
from volume import *
from mathutils import Vector, Quaternion
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap


def run_retarget_mesh(
    config,
    src_path,
    dest_path,
    out_path,
    use_cuda=True,
    batch_size=1,
    visualize=False
) -> Dict[str, Tensor]:
    
    ##############################################################
    ##                  Parameter setting                       ##
    ##############################################################

    assert batch_size == 1, 'PyTorch L-BFGS only supports batch_size == 1'
    device = torch.device('cuda') if use_cuda else torch.device('cpu')

    ''' 
        Parameters weighting (g: gamma)
        Energy term: gshape, gvol, gc
        Contact term: gr, ga
        Ground contact: grg, gag
        Offset weight: epsilon
    '''
    gshape = gvol = gc = 1
    gr = ga = 1
    grg = gag = 0.1
    eps = 0.3

    ##############################################################
    ##                   Mesh retargeting                       ##
    ##############################################################
    
    # vsource = source_mesh['vertices']
    # vsoutceStd = source_std_mesh['vertices']
    # vtarget = target_mesh['vertices']
    # fsource = source_mesh['faces']
    # fsourceStd = source_std_mesh['faces']
    # ftarget = target_mesh['faces']

    # smpl_segm = getPartSegm(config)

    # source_height = getHight(vsoutceStd)
    # target_height = getHight(vtarget)
    # vsource = setScale(vsource, target_height/source_height)

    # nbSource = getNeighbors(vsource, fsource)
    # nbTarget = getNeighbors(vtarget, ftarget)

    # offsetTarget = getLaplacianOffset(vtarget, nbTarget)
    # target_vol = getVolumes(vtarget, ftarget, smpl_segm)
    
    # for i in range(1):
    #     offsetSource = getLaplacianOffset(vsource, nbSource)
    #     shapeDirection = getShapeDirection(vsource, nbSource, offsetSource, offsetTarget)
    #     EShape = getShapeEnergy(offsetSource, offsetTarget)

    #    source_vol = getVolumes(vsource, fsource, smpl_segm)
    #    volumeDirection = getVolumeDirection(source_vol, target_vol, offsetSource, smpl_segm)
    #    EVol = getVolumeEnergy(source_vol, target_vol)
        
    #    vsource += eps*(shapeDirection)

    # if visualize:
    #     mesh = trimesh.Trimesh(vsource, fsource, process=False)
    #     mesh.show()

def setScale(vsource, scale):
    vsource *= scale
    return vsource

def getHight(vertices):
    return max(vertices[:,1]) - min(vertices[:,1])

def getShapeEnergy(offsetSource, offsetTarget):
    EShape = np.zeros(3)
    for i in range(len(offsetSource)):
        EShape += (offsetSource[i] - offsetTarget[i])**2
    return EShape

def getVolumeEnergy(volumeSource, volumeTarget):
    EVol = 0
    for part in volumeSource:
        EVol += (volumeSource[part] - volumeTarget[part])**2
    return EVol