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

from tkinter import *
from tqdm import tqdm
from loguru import logger
from typing import Optional, Dict, Callable, Union
from data import mesh
from utils import (Tensor)
from human_body_prior.tools.model_loader import load_vposer
from shape import *
from volume import *

def run_retarget_mesh(
    config,
    src_mesh,
    src_std_mesh,
    dest_mesh,
    out_path,
    batch_size=1,
    visualize=False
) -> Dict[str, Tensor]:
    
    ##############################################################
    ##                  Parameter setting                       ##
    ##############################################################

    # assert batch_size == 1, 'PyTorch L-BFGS only supports batch_size == 1'
    # device = torch.device('cuda') if use_cuda else torch.device('cpu')

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
    
    # Get vertices and faces of meshs
    src_vertice = src_mesh['vertices']; src_face = src_mesh['faces']
    src_std_vertice = src_std_mesh['vertices']; src_std_face = src_std_mesh['faces']
    dest_vertice = dest_mesh['vertices']; dest_face = dest_mesh['faces']

    # Get body segmentation
    smpl_segm = getPartSegm(config)

    # Set scale of source vertice
    src_height = getHight(src_std_vertice)
    dest_height = getHight(dest_vertice)
    src_vertice = setScale(src_vertice, dest_height/src_height)

    src_neighbor = getNeighbors(src_vertice, src_face)
    dest_neighbor = getNeighbors(dest_vertice, dest_face)

    dest_offset = getLaplacianOffset(dest_vertice, dest_neighbor)
    # dest_vol = getVolumes(dest_vertice, dest_face, smpl_segm)

    for i in range(1):
        src_offset = getLaplacianOffset(src_vertice, src_neighbor)
        shape_direction = getShapeDirection(src_vertice, src_neighbor, src_offset, dest_offset)
        shape_energy = getShapeEnergy(src_offset, dest_offset)

        # src_vol = getVolumes(src_vertice, src_face, smpl_segm)
        # volume_direction = getVolumeDirection(src_vol, dest_vol, src_offset, smpl_segm)
        # volume_energy = getVolumeEnergy(src_vol, dest_vol)
        
        src_vertice += eps*(shape_direction)

    if True:
        mesh = trimesh.Trimesh(src_vertice, src_face, process=False)
        mesh.show()

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