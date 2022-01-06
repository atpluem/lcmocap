import sys
from networkx.algorithms.cluster import triangles
import numpy as np
from pyrender import light
import torch
import torch.nn as nn
import smplifyx.fitting as fitting
import trimesh
import networkx as nx
import json

from tkinter import *
from tqdm import tqdm
from loguru import logger
from typing import Optional, Dict, Callable, Union
from data import mesh
from utils import (Tensor)
from human_body_prior.tools.model_loader import load_vposer
from smplifyx.mesh_viewer import MeshViewer as mv
from shape import *
from volume import *

def run_retarget(
    config,
    source_mesh,
    source_std_mesh,
    target_mesh,
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
    ##                  Local Shape Fidelity                    ##
    ##############################################################
    
    # nbSource = getNeighbors(source_mesh['vertices'], source_mesh['faces'])
    # nbTarget = getNeighbors(target_mesh['vertices'], target_mesh['faces'])
    # offsetSource = getLaplacianOffset(source_mesh['vertices'], nbSource)
    # offsetTarget = getLaplacianOffset(target_mesh['vertices'], nbTarget)
    
    # EShape = getShapeEnergy(offsetSource, offsetTarget)
    # shapeDirection = getShapeDirection(vsource, nbSource, offsetSource, offsetTarget)
    

    ##############################################################
    ##                  Volume Preservation                     ##
    ##############################################################

    # smpl_segm = getPartSegm(config)
    # source_vol = getVolumes(config, source_mesh['vertices'], source_mesh['faces'], smpl_segm)
    # target_vol = getVolumes(config, target_mesh['vertices'], target_mesh['faces'], smpl_segm)

    # EVol = getVolumeEnergy(source_vol, target_vol)
    # volumeDirection = getVolumeDirection(source_vol, target_vol, offsetSource, smpl_segm)


    ##############################################################
    ##                      Contacts                            ##
    ##############################################################


    ##############################################################
    ##                    Mesh Rotation                         ##
    ##############################################################
    
    # from numpy import sin, cos
    # from itertools import combinations, product
    # import matplotlib.pyplot as plt
    # theta = np.radians(30)
    # d = [-2, 2]
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.set_aspect("auto")
    # ax.set_autoscale_on(True)

    # for id in smpl_segm['head']:
    #     vsource[id] = [vsource[id][0]*cos(theta) - vsource[id][1]*sin(theta),
    #                    vsource[id][0]*sin(theta) + vsource[id][1]*cos(theta),
    #                    vsource[id][2]] 
    # plt.show()

    ##############################################################
    ##                 Iterative Solving                        ##
    ##############################################################

    vsource = source_mesh['vertices']
    vsoutceStd = source_std_mesh['vertices']
    vtarget = target_mesh['vertices']
    fsource = source_mesh['faces']
    ftarget = target_mesh['faces']
    
    smpl_segm = getPartSegm(config)

    source_height = getHight(vsoutceStd)
    target_height = getHight(vtarget)
    vsource = setScale(vsource, target_height/source_height)

    nbSource = getNeighbors(vsource, fsource)
    nbTarget = getNeighbors(vtarget, ftarget)

    offsetTarget = getLaplacianOffset(vtarget, nbTarget)
    target_vol = getVolumes(vtarget, ftarget, smpl_segm)
    
    # for i in range(1):
    #     offsetSource = getLaplacianOffset(vsource, nbSource)
    #     shapeDirection = getShapeDirection(vsource, nbSource, offsetSource, offsetTarget)
    #     EShape = getShapeEnergy(offsetSource, offsetTarget)

    #     source_vol = getVolumes(vsource, fsource, smpl_segm)
    #     volumeDirection = getVolumeDirection(source_vol, target_vol, offsetSource, smpl_segm)
    #     EVol = getVolumeEnergy(source_vol, target_vol)

    #     vsource += eps*(shapeDirection + volumeDirection)

    if visualize:
        mesh = trimesh.Trimesh(vsource, fsource, process=False)
        mesh.show()
        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax = Axes3D(fig)
        # ax = ax.plot_trisurf(vsource[:,0], vsource[:,1], triangles=fsource, Z=vsource[:,2])
        # plt.show()

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