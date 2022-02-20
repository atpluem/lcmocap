import enum
import imp
import sys
from tkinter.tix import Tree
from traceback import print_tb
from unittest import result
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
import collections
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
    
    ERIC_JOINT_NAMES = ['root', 'upperleg_l', 'upperleg_r', 'spine_01', 
    'lowerleg_l', 'lowerleg_r', 'spine_02', 'foot_l', 'foot_r',
    'spine_03', 'ball_l', 'ball_r', 'neck', 'shoulder_l', 'shoulder_r',
    'head', 'upperarm_l', 'upperarm_r', 'lowerarm_l', 'lowerarm_r',
    'hand_l', 'hand_r']

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
    ##              Initial armature and pose                   ##
    ##############################################################

    '''
        Load FBX file of source and dstination
        Change name of armature
        Keep only armature,then delete other objects (etc. light, camera)
    '''
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

    '''
        Initial source and destination pose by mapping body_pose
    '''

    # Source armature
    src_joints = {}
    part_idx = 0
    for bone in bpy.data.objects['SRC'].data.bones:
        if bone.name in SMPLX_JOINT_NAMES and part_idx < NUM_SMPLX_BODYJOINTS:
            if bone.name == 'pelvis': 
                src_joints[bone.name] = np.array(bpy.data.objects['SRC'].pose.bones[bone.name].head)
                continue
            set_pose(bpy.data.objects['SRC'], SMPLX_JOINT_NAMES[part_idx+1], body_pose[part_idx])
            bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
            src_joints[bone.name] = np.array(bpy.data.objects['SRC'].pose.bones[bone.name].head)
            part_idx = part_idx + 1

    # Destination armature
    dest_joints = {}
    part_idx = 0
    for bone in bpy.data.objects['DEST'].data.bones:
        if bone.name in ERIC_JOINT_NAMES and part_idx < NUM_SMPLX_BODYJOINTS:
            # Rename joints to default name
            idx = ERIC_JOINT_NAMES.index(bone.name)
            bpy.data.objects['DEST'].pose.bones[bone.name].name = SMPLX_JOINT_NAMES[idx]
            if bpy.data.objects['DEST'].pose.bones[bone.name].name == 'pelvis': 
                dest_joints[bone.name] = np.array(bpy.data.objects['DEST'].pose.bones[bone.name].head)
                continue
            # set_pose(bpy.data.objects['DEST'], bpy.data.objects['DEST'].pose.bones[bone.name].name, body_pose[idx-1])
            # bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
            dest_joints[bone.name] = np.array(bpy.data.objects['DEST'].pose.bones[bone.name].head)
            part_idx = part_idx + 1

    ##############################################################
    ##         Create dataframe for joint location              ##
    ##############################################################

    # Sort joints by key
    src_joints = collections.OrderedDict(sorted(src_joints.items()))
    dest_joints = collections.OrderedDict(sorted(dest_joints.items()))

    # Reshape into (N, 3)
    src_coor = np.concatenate(list(src_joints.values()), axis=0).reshape((-1,3))
    dest_coor = np.concatenate(list(dest_joints.values()), axis=0).reshape((-1,3))

    df = pd.DataFrame(src_joints.keys(), columns=['joint'])
    df['src_x'] = src_coor[:,0] * 100 
    df['src_y'] = src_coor[:,2] * 100 # y-coor at position index 2
    df['src_z'] = src_coor[:,1] * -100 # z-coor at position index 1
    df['dest_x'] = dest_coor[:,0]
    df['dest_y'] = dest_coor[:,1]
    df['dest_z'] = dest_coor[:,2]

    # Define part of body
    Spine = ['pelvis', 'spine1', 'spine2', 'spine3', 'neck', 'head']
    LeftArm = ['left_collar', 'left_shoulder', 'left_elbow', 'left_wrist']
    RightArm = ['right_collar', 'right_shoulder', 'right_elbow', 'right_wrist']
    LeftLeg = ['left_hip', 'left_knee', 'left_ankle', 'left_foot']
    RightLeg = ['right_hip', 'right_knee', 'right_ankle', 'right_foot']
    Root = ['pelvis', 'spine1', 'left_hip', 'right_hip']

    df.loc[df['joint'].isin(Spine), 'part'] = 'Spine'
    df.loc[df['joint'].isin(LeftArm), 'part'] = 'LeftArm'
    df.loc[df['joint'].isin(RightArm), 'part'] = 'RightArm'
    df.loc[df['joint'].isin(LeftLeg), 'part'] = 'LeftLeg'
    df.loc[df['joint'].isin(RightLeg), 'part'] = 'RightLeg'

    ##############################################################
    ##                   Calculate error                        ##
    ##############################################################

    # Calculate source pose distance
    miss = {}
    pelvis = df[df['joint'] == 'pelvis']
    for idx, part in df.iterrows():
        src_dist = np.array([math.sqrt((part['src_x'] - pelvis['src_x'].values)**2 + (part['src_y'] - pelvis['src_y'].values)**2), \
                            math.sqrt((part['src_x'] - pelvis['src_x'].values)**2 + (part['src_z'] - pelvis['src_z'].values)**2), \
                            math.sqrt((part['src_z'] - pelvis['src_z'].values)**2 + (part['src_y'] - pelvis['src_y'].values)**2)]) 

    # Try to adjust destination pose
    poses = {}
    for part in LeftArm:
        lr = 0.05
        min_loss = [1000,1000,1000]
        state = 0
        pose = [0,0,0]
        if part in Root:
            continue
        print(part)
        while True:
            if state == 0:
                pose[0] = pose[0] + lr
            elif state == 1:
                pose[1] = pose[1] + lr
            elif state == 2:
                pose[2] = pose[2] + lr
            
            set_pose(bpy.data.objects['DEST'], LeftArm[LeftArm.index(part)-1], pose)
            bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

            # Calculate distances of destination
            for child in LeftArm:              
                df.loc[df['joint'] == child, ['dest_x', 'dest_y', 'dest_z']] = \
                    np.array(bpy.data.objects['DEST'].pose.bones[child].head)
                part_df = df.loc[df['joint'] == part]
                dest_dist = np.array([math.sqrt((part_df['dest_x'] - pelvis['dest_x'].values)**2 + (part_df['dest_y'] - pelvis['dest_y'].values)**2), \
                                    math.sqrt((part_df['dest_x'] - pelvis['dest_x'].values)**2 + (part_df['dest_z'] - pelvis['dest_z'].values)**2), \
                                    math.sqrt((part_df['dest_z'] - pelvis['dest_z'].values)**2 + (part_df['dest_y'] - pelvis['dest_y'].values)**2)])
            
            loss = abs(dest_dist - src_dist)
            if loss[0] < min_loss[0] or loss[1] < min_loss[1] or loss[2] < min_loss[2]:
                min_loss = loss
                print(min_loss)
            else :
                state = state + 1     
            
            if state == 4: break
        poses[part] = pose
    
    print(poses)

    # Visualize
    fig, axes = plt.subplots(2, 3)
    sns.scatterplot(ax=axes[0,0], data=df, x='src_x', y='src_y', hue='part')
    sns.scatterplot(ax=axes[0,1], data=df, x='src_z', y='src_y', hue='part')
    sns.scatterplot(ax=axes[0,2], data=df, x='src_x', y='src_z', hue='part')
    sns.scatterplot(ax=axes[1,0], data=df, x='dest_x', y='dest_y', hue='part')
    sns.scatterplot(ax=axes[1,1], data=df, x='dest_z', y='dest_y', hue='part')
    sns.scatterplot(ax=axes[1,2], data=df, x='dest_x', y='dest_z', hue='part')
    plt.show()

    # Export pkl
    # result = {}
    # result["body_pose"] = body_pose
    # print(out_path)
    # output = open(out_path+'/retar.pkl', 'wb')
    # pickle.dump(result, output)
    # output.close()

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