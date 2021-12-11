import sys
import numpy as np
from numpy.core.fromnumeric import mean
from numpy.lib.arraysetops import unique
from numpy.lib.twodim_base import tri
from numpy.lib.utils import source
from pyrender import light
import torch
import torch.nn as nn
import fitting
import trimesh
import networkx as nx
import json

from tqdm import tqdm
from loguru import logger
from typing import Optional, Dict, Callable, Union
from data import mesh
from utils import (Tensor)
from human_body_prior.tools.model_loader import load_vposer
from mesh_viewer import MeshViewer as mv
from shape import getShapeOffset
from volume import getVolumes

def run_retarget(
    config,
    source_mesh,
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

    # with fitting.FittingMonitor() as monitor:
        # monitor.run_fitting()
        # monitor.get_LaplacianMatrixUmbrella(source_vertices, source_faces)
        # monitor.get_LaplacianMatrixCotangent(source_vertices, source_faces)

    ##############################################################
    ##                  Local Shape Fidelity                    ##
    ##############################################################
    

    # offsetSource = getShapeOffset(source_mesh['vertices'], source_mesh['faces'])
    # offsetTarget = getShapeOffset(source_mesh['vertices'], source_mesh['faces'])
    
    # EShape = getShapeEnergy(offsetSource, offsetTarget)
    # vHat = getOptimalPosition(source_mesh['vertices'], source_nb, offsetSource, offsettarget)


    ##############################################################
    ##                  Volume Preservation                     ##
    ##############################################################


    source_vol = getVolumes(config, source_mesh['vertices'], source_mesh['faces'])
    target_vol = getVolumes(config, target_mesh['vertices'], target_mesh['faces'])

    EVol = getVolumeEnergy(source_vol, target_vol)
    print(EVol)

    ##############################################################
    ##                      Contacts                            ##
    ##############################################################

    

    ##############################################################
    ##                 Iterative Solving                        ##
    ##############################################################


    if visualize:
        mesh = trimesh.Trimesh(source_mesh['vertices'], 
                               source_mesh['faces'], process=False)

        mesh.show()

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