import enum
import imp
import sys
from tkinter.tix import Tree
from traceback import print_tb
from matplotlib import markers, projections
from matplotlib.pyplot import axis
from mpl_toolkits.mplot3d import Axes3D
from networkx.algorithms.cluster import triangles
import numpy as np
from pyrender import light
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


def run_retarget(
    config,
    pkl,
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

    # หมุนกระดูกไปทีละนิด แล้วเปรียบเทียบท่าทางว่าเหมือนกันไหม
    # Convert any model to SMPL for compare right pose
    # group part of body then check pose

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

    ##############################################################
    ##                  Read PKL for rotation                   ##
    ##############################################################
    
    NUM_SMPLX_BODYJOINTS=21
    SMPLX_JOINT_NAMES = ['pelvis', 'left_hip', 'right_hip', 'spine1', 
    'left_knee', 'right_knee', 'spine2', 'left_ankle', 'right_ankle', 
    'spine3', 'left_foot', 'right_foot', 'neck', 'left_collar', 'right_collar', 
    'head', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 
    'left_wrist', 'right_wrist']

    SMPL_JOINT_NAMES = ['pelvis', 'L_Hip', 'R_Hip', 'Spine1', 
    'L_Knee', 'R_Knee', 'Spine2', 'L_Ankle', 'R_Ankle', 
    'Spine3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 
    'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 
    'L_Wrist', 'R_Wrist']

    AJ_JOINT_NAMES = ['Hips', 'LeftUpLeg', 'RightUpLeg', 'Spine',
    'LeftLeg', 'RightLeg', 'Spine1', 'LeftFoot', 'RightFoot',
    'Spine2', 'LeftToeBase', 'RightToeBase', 'Neck', 'LeftShoulder', 'RightShoulder',
    'Head', 'LeftArm', 'RightArm', 'LeftForeArm', 'RightForeArm',
    'LeftHand', 'RightHand']

    with open(pkl, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    if "transl" in data:
        translation = np.array(data["transl"]).reshape(3)

    if "global_orient" in data:
        global_orient = np.array(data["global_orient"]).reshape(3)

    body_pose = np.array(data["body_pose"])
    
    if body_pose.shape != (1, NUM_SMPLX_BODYJOINTS * 3):
        print(f"Invalid body pose dimensions: {body_pose.shape}")
        body_data = None

    body_pose = np.array(data["body_pose"]).reshape(NUM_SMPLX_BODYJOINTS, 3)
    jaw_pose = np.array(data["jaw_pose"]).reshape(3)
    left_hand_pose = np.array(data["left_hand_pose"]).reshape(-1, 3)
    right_hand_pose = np.array(data["right_hand_pose"]).reshape(-1, 3)

    betas = np.array(data["betas"]).reshape(-1).tolist()
    expression = np.array(data["expression"]).reshape(-1).tolist()

    ##############################################################
    ##                  Skeleton retargeting                    ##
    ##############################################################

    # Load FBX
    bpy.ops.import_scene.fbx(filepath=src_path)
    bpy.ops.object.select_all(action='DESELECT')
    for ob in bpy.data.objects:
        if ob.type != 'ARMATURE':
            bpy.data.objects[ob.name].select_set(True)
        elif ob.type == 'ARMATURE' and ob.name != 'SRC':
            bpy.data.objects[ob.name].name = 'SRC'

    bpy.ops.import_scene.fbx(filepath=dest_path)
    bpy.ops.object.select_all(action='DESELECT')
    for ob in bpy.data.objects:
        if ob.type != 'ARMATURE':
            bpy.data.objects[ob.name].select_set(True)
        elif ob.type == 'ARMATURE' and ob.name != 'SRC' and ob.name != 'DEST':
            bpy.data.objects[ob.name].name = 'DEST'
    
    bpy.ops.object.delete()

    # Source armature
    src_joints = {}
    for i, bone in enumerate(bpy.data.objects['SRC'].data.bones):
        if bone.name in SMPLX_JOINT_NAMES:
            src_joints[bone.name] = np.array(bpy.data.objects['SRC'].pose.bones[bone.name].center) 

    # Destination armature
    dest_joints = {}
    for i, bone in enumerate(bpy.data.objects['DEST'].data.bones):
        if bone.name in AJ_JOINT_NAMES:
            idx = AJ_JOINT_NAMES.index(bone.name)
            bpy.data.objects['DEST'].pose.bones[bone.name].name = SMPLX_JOINT_NAMES[idx]
            dest_joints[bone.name] = np.array(bpy.data.objects['DEST'].pose.bones[bone.name].center)

    # Create dataframe
    src_coor = np.concatenate(list(src_joints.values()), axis=0).reshape((-1,3))
    dest_coor = np.concatenate(list(dest_joints.values()), axis=0).reshape((-1,3))
    
    df = pd.DataFrame(src_joints.keys(), columns=['joint'])
    df['src_x'] = src_coor[:,0]
    df['src_y'] = src_coor[:,1]
    df['src_z'] = src_coor[:,2]
    df['dest_x'] = dest_coor[:,0]
    df['dest_y'] = dest_coor[:,1]
    df['dest_z'] = dest_coor[:,2]

    # Define part of body
    Spine = ['pelvis', 'spine1', 'spine2', 'spine3', 'neck', 'head']
    LeftArm = ['left_collar', 'left_shoulder', 'left_elbow', 'left_wrist']
    RightArm = ['right_collar', 'right_shoulder', 'right_elbow', 'right_wrist']
    LeftLeg = ['left_hip', 'left_knee', 'left_ankle', 'left_foot']
    RightLeg = ['right_hip', 'right_knee', 'right_ankle', 'right_foot']

    df.loc[df['joint'].isin(Spine), 'part'] = 'Spine'
    df.loc[df['joint'].isin(LeftArm), 'part'] = 'LeftArm'
    df.loc[df['joint'].isin(RightArm), 'part'] = 'RightArm'
    df.loc[df['joint'].isin(LeftLeg), 'part'] = 'LeftLeg'
    df.loc[df['joint'].isin(RightLeg), 'part'] = 'RightLeg'

    # minmax scaling
    print(df)

    # Find Error
    miss = {}
    hip = df[df['joint'] == 'pelvis']
    # lh = df[df['joint'] == 'LeftHand']
    for idx, part in df.iterrows():
        if not (part['joint'] in Spine):
            src_dist = np.array([math.sqrt((part['src_x'] - hip['src_x'].values)**2 + (part['src_y'] - hip['src_y'].values)**2), \
                                math.sqrt((part['src_x'] - hip['src_x'].values)**2 + (part['src_z'] - hip['src_z'].values)**2), \
                                math.sqrt((part['src_z'] - hip['src_z'].values)**2 + (part['src_y'] - hip['src_y'].values)**2)])
            dest_dist = np.array([math.sqrt((part['dest_x'] - hip['dest_x'].values)**2 + (part['dest_y'] - hip['dest_y'].values)**2), \
                                 math.sqrt((part['dest_x'] - hip['dest_x'].values)**2 + (part['dest_z'] - hip['dest_z'].values)**2), \
                                 math.sqrt((part['dest_z'] - hip['dest_z'].values)**2 + (part['dest_y'] - hip['dest_y'].values)**2)])
            miss[part['joint']] =  abs(dest_dist - src_dist)

    fig, axes = plt.subplots(1,2)
    sns.scatterplot(ax=axes[0], data=df, x='src_y', y='src_z', hue='part')
    sns.scatterplot(ax=axes[1], data=df, x='dest_y', y='dest_z', hue='part')
    plt.show()

    # bpy.ops.export_scene.fbx(filepath=out_path+'/retar.fbx', use_selection=False)

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

def set_pose(armature, bone_name, rodrigues, rodrigues_ref=None):
    rod = Vector((rodrigues[0], rodrigues[1], rodrigues[2]))
    angle_rad = rod.length
    axis = rod.normalized()

    if armature.pose.bones[bone_name].rotation_mode != 'QUATERNION':
        armature.pose.bones[bone_name].rotation_mode = 'QUATERNION'

    quat = Quaternion(axis, angle_rad)

    if rodrigues_ref is None:
        armature.pose.bones[bone_name].rotation_quaternion = quat
    else:
        rod_ref = Vector((rodrigues[0], rodrigues[1], rodrigues[2]))
        rod_result = rod + rod_ref
        angle_rad_result = rod_result.length
        axis_result = rod_result.normalized()
        quat_result = Quaternion(axis_result, angle_rad_result)
        armature.pose.bones[bone_name].rotation_quaternion = quat_result
    return