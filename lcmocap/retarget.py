import sys
import numpy as np
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

    vertices = source_mesh['vertices']
    faces = source_mesh['faces']

    ##############################################################
    ##                  Local Shape Fidelity                    ##
    ##############################################################

    offsetSource = getLaplacianOffset(source_mesh['vertices'], source_mesh['faces'])
    offsetTarget = getLaplacianOffset(target_mesh['vertices'], target_mesh['faces'])

    EShape = getShapeEnergy(offsetSource, offsetTarget)
    vHat = getOptimalPosition(source_mesh['vertices'], source_mesh['faces'],
                                    offsetSource, offsetTarget)


    ##############################################################
    ##                  Volume Preservation                     ##
    ##############################################################

    ''' SMPL body part segmentation (24 parts 7390 unit)
        -----
        leftHand        324 rightHand       324
        leftUpLeg       254 rightUpLeg      254 // 
        leftArm         284 rightArm        284
        leftLeg         217 rightLeg        217 //
        leftToeBase     122 rightToeBase    122 
        leftFoot        143 rightFoot       143
        leftShoulder    151 rightShoulder   151
        leftHandIndex1  478 rightHandIndex1 478
        leftForeArm     246 rightForeArm    246 //
        spine           233     //
        spine1          267     //
        spine2          615     //
        head            1194
        neck            156
        hips            487     //

        Paper 17 parts
        -----
        head + neck     Hand + HandIndex1
        Shoulder + Arm  Foot + ToeBase
    '''
    # Read JSON segmentation file
    segm_path = config.body_part_segm.smpl_path
    with open(segm_path) as json_file:
        smpl_segm = json.load(json_file)

    # Union parts (24 -> 17)
    head_segm = list(set.union(set(smpl_segm['head']), set(smpl_segm['neck'])))
    leftHand_segm = list(set.union(set(smpl_segm['leftHand']), set(smpl_segm['leftHandIndex1'])))
    rightHand_segm = list(set.union(set(smpl_segm['rightHand']), set(smpl_segm['rightHandIndex1'])))
    leftArm_segm = list(set.union(set(smpl_segm['leftShoulder']), set(smpl_segm['leftArm'])))
    rightArm_segm = list(set.union(set(smpl_segm['rightShoulder']), set(smpl_segm['rightArm'])))
    leftFoot_segm = list(set.union(set(smpl_segm['leftFoot']), set(smpl_segm['leftToeBase'])))
    rightFoot_segm = list(set.union(set(smpl_segm['rightFoot']), set(smpl_segm['rightToeBase'])))

    # Create seams
    leftArm_coor = getSeamCoor(vertices, faces, list(set(leftArm_segm) & set(smpl_segm['leftForeArm'])))
    


    ##############################################################
    ##                      Contacts                            ##
    ##############################################################



    ##############################################################
    ##                 Iterative Solving                        ##
    ##############################################################

    # newVertices = vertices + eps*(vHat)

    if visualize:
        mesh = trimesh.Trimesh(vertices, 
                               source_mesh['faces'], process=False)

        mesh.show()


def getLaplacianOffset(vertices, faces):
        
    N = vertices.shape[0]
    M = faces.shape[0] 

    mesh = trimesh.Trimesh(vertices, faces)
    graph = nx.from_edgelist(mesh.edges_unique)
    neighbors = [list(graph[i].keys()) for i in range(N)]    
    neighbors = np.array(neighbors)
    
    offset = np.zeros((N, 3))
    for i in range(N):
        indice = len(neighbors[i])
        for j in range(indice):
            offset[i] = offset[i] + (vertices[neighbors[i][j]]/indice - vertices[i])

    return offset
    
def getShapeEnergy(offsetSource, offsetTarget):

    EShape = []
    for i in range(offsetSource.shape[0]):
        EShape = (offsetSource[i] - offsetTarget)**2

    return EShape

def getOptimalPosition(vertices, faces, offsetSource, offsetTarget):
    
    N = vertices.shape[0]
    mesh = trimesh.Trimesh(vertices, faces)
    graph = nx.from_edgelist(mesh.edges_unique)
    neighbors = [list(graph[i].keys()) for i in range(N)]    
    neighbors = np.array(neighbors)

    newVertices = np.zeros((N, 3))
    for i in range(N):
        indice = len(neighbors[i])
        for j in range(indice):
            newVertices[i] = newVertices[i] + (vertices[neighbors[i][j]]/indice) \
                            - (   offsetTarget[i][0]*offsetSource[i][0] \
                                + offsetTarget[i][1]*offsetSource[i][1] \
                                + offsetTarget[i][2]*offsetSource[i][2])

    return newVertices

def getSeamCoor(vertices, faces, vseam):

    # coordinate of centroid
    centroid = np.zeros(3)
    triangles = []

    # Find centroid and neighbors between each vertices
    for count, vertex in enumerate(vseam):
        centroid[0] += vertices[vertex][0]
        centroid[1] += vertices[vertex][1]
        centroid[2] += vertices[vertex][2]

        neighbors = set()
        # Find neighbors in all triangle face
        for triangle in faces:
            if vertex in triangle and len(set(vseam) & set(triangle)) == 2:
                neighbors.update(set(vseam) & set(triangle))

        # Create triangle of seam 
        for count, v in enumerate(neighbors):
            tri_seam = []
            if v != vertex:
                tri_seam.append(vertex)
                tri_seam.append(v)
                tri_seam.append(-1)
                triangles.append(tri_seam)
    
    tri_list = list(set([tuple(sorted(x)) for x in triangles]))
    centroid /= len(vseam)

    # Get coordinate from vertice id
    coor = []
    for triangle in tri_list:
        point = []
        point.append(centroid)
        point.append(vertices[triangle[1]])
        point.append(vertices[triangle[2]])
        coor.append(np.array(point))

    return np.array(coor)
        
def getSignedVolume(pi, pj, pk):
    vkji = pk[0]*pj[1]*pi[2]
    vjki = pj[0]*pk[1]*pi[2]
    vkij = pk[0]*pi[1]*pj[2]
    vikj = pi[0]*pk[1]*pj[2]
    vjik = pj[0]*pi[1]*pk[2]
    vijk = pi[0]*pj[1]*pk[2]
    return (1.0/6.0) * (-vkji+vjki+vkij-vikj-vjik+vijk)

def getMeshVolume():
    print(1)
