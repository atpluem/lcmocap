import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import collections
import seaborn as sns
import bpy

from typing import Optional, Dict, Callable, Union
from utils import (Tensor)
from mathutils import Vector, Quaternion, Matrix
from utils.utilfuncs import *
from numpy.random import randint, rand
from genetic_algo import get_pose_ga
from euler_rotate import get_pose_euler
from quaternion_rotate import get_pose_quaternion
from sklearn.decomposition import PCA

def run_retarget(
    config,
    pose_params_path,
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

    PEGGY_JOINT_NAMES = []

    JOINTS = config.datasets.joints

    print(pose_params_path)
    with open(pose_params_path, 'rb') as f:
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
        Load FBX file of source and destination
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

    # Destination armature
    dest_joints = {}
    dest_orien = {}
    part_idx = 0
    for bone in bpy.data.objects['DEST'].data.bones:
        if bone.name in JOINTS and part_idx < NUM_SMPLX_BODYJOINTS:
            # Rename joints to default name
            idx = JOINTS.index(bone.name)
            bpy.data.objects['DEST'].pose.bones[bone.name].name = SMPLX_JOINT_NAMES[idx]
            # Check bone orientation
            orien = abs(np.array(bpy.data.objects['DEST'].pose.bones[bone.name].head - \
                                 bpy.data.objects['DEST'].pose.bones[bone.name].tail))
            if np.argmax(orien) == 2:
                dest_orien[bone.name] = 'h' # horizontal joint
            elif np.argmax(orien) == 1:
                dest_orien[bone.name] = 'v' # virtical joint
            # root joint condition
            if bpy.data.objects['DEST'].pose.bones[bone.name].name == 'pelvis': 
                dest_joints[bone.name] = np.array(bpy.data.objects['DEST'].pose.bones[bone.name].head)
                continue
            dest_joints[bone.name] = np.array(bpy.data.objects['DEST'].pose.bones[bone.name].head)
            part_idx = part_idx + 1

    # Source armature
    ''' SMPLX base '''
    # src_joints = {}
    # part_idx = 0
    # for bone in bpy.data.objects['SRC'].data.bones:
    #     if bone.name in SMPLX_JOINT_NAMES and part_idx < NUM_SMPLX_BODYJOINTS:
    #         # root joint condition
    #         if bone.name == 'pelvis':
    #             src_joints[bone.name] = np.array(bpy.data.objects['SRC'].pose.bones[bone.name].head)
    #             continue
    #         src_joints[bone.name] = np.array(bpy.data.objects['SRC'].pose.bones[bone.name].head)
    #         part_idx = part_idx + 1

    ''' AJ base '''
    src_joints = {}
    part_idx = 0
    for bone in bpy.data.objects['SRC'].data.bones:
        if bone.name in AJ_JOINT_NAMES and part_idx < NUM_SMPLX_BODYJOINTS:
            idx = AJ_JOINT_NAMES.index(bone.name)
            bpy.data.objects['SRC'].pose.bones[bone.name].name = SMPLX_JOINT_NAMES[idx]
            # root joint condition
            if bpy.data.objects['SRC'].pose.bones[bone.name].name == 'pelvis':
                src_joints[bone.name] = np.array(bpy.data.objects['SRC'].pose.bones[bone.name].head)
                continue
            src_joints[bone.name] = np.array(bpy.data.objects['SRC'].pose.bones[bone.name].head)
            part_idx = part_idx + 1

    '''
        Calculate new Domain of both Riggings
    '''
    scales = new_domain(bpy.data.objects['SRC'], bpy.data.objects['DEST'])

    # Scale source armature,then initial pose of both armatures 
    body_scale = dest_joints['head'][1] / src_joints['head'][1]
    for idx, body_parts in enumerate(SMPLX_JOINT_NAMES):
        # bpy.data.objects['SRC'].pose.bones[body_parts].scale *= scale
        if bpy.data.objects['SRC'].pose.bones[body_parts].name == 'pelvis':
            src_joints[body_parts] = np.array(bpy.data.objects['SRC'].pose.bones[body_parts].head)
            continue
        # Set the pose according to pose parameter
        set_pose(bpy.data.objects['SRC'], bpy.data.objects['SRC'].pose.bones[body_parts].name, body_pose[idx-1])
        # if dest_orien[body_parts] == 'v':
        #     poses[body_parts] = [body_pose[idx-1,0],body_pose[idx-1,1],body_pose[idx-1,2]]
        # elif dest_orien[body_parts] == 'h':
        #     poses[body_parts] = [body_pose[idx-1,0],body_pose[idx-1,1],body_pose[idx-1,2]]
        # set_pose(bpy.data.objects['DEST'], bpy.data.objects['DEST'].pose.bones[body_parts].name, poses[body_parts])
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        src_joints[body_parts] = np.array(bpy.data.objects['SRC'].pose.bones[body_parts].head)
        dest_joints[body_parts] = np.array(bpy.data.objects['DEST'].pose.bones[body_parts].head)

    ##############################################################
    ##         Create dataframe for joint's location            ##
    ##############################################################

    # Sort joints by key
    src_joints = collections.OrderedDict(sorted(src_joints.items()))
    dest_joints = collections.OrderedDict(sorted(dest_joints.items()))
    dest_orien = collections.OrderedDict(sorted(dest_orien.items()))

    # Reshape into (N, 3)
    src_coor = np.concatenate(list(src_joints.values()), axis=0).reshape((-1,3))
    dest_coor = np.concatenate(list(dest_joints.values()), axis=0).reshape((-1,3))
    dest_orien = list(dest_orien.values())

    df = pd.DataFrame(src_joints.keys(), columns=['joint'])
    df['src_x'] = src_coor[:,0]
    df['src_y'] = src_coor[:,1]
    df['src_z'] = src_coor[:,2]
    # df['src_y'] = src_coor[:,2] * 100 # y-coor at position index 2
    # df['src_z'] = src_coor[:,1] * -100 # z-coor at position index 1
    df['dest_x'] = dest_coor[:,0]
    df['dest_y'] = dest_coor[:,1]
    df['dest_z'] = dest_coor[:,2]
    df['dest_orien'] = dest_orien

    # Define part of body
    Spine = ['pelvis', 'spine1', 'spine2', 'spine3', 'neck', 'head']
    LeftArm = ['left_collar', 'left_shoulder', 'left_elbow', 'left_wrist']
    RightArm = ['right_collar', 'right_shoulder', 'right_elbow', 'right_wrist']
    LeftLeg = ['left_hip', 'left_knee', 'left_ankle', 'left_foot']
    RightLeg = ['right_hip', 'right_knee', 'right_ankle', 'right_foot']
    body_segm = dict()
    body_segm['spine'] = Spine
    body_segm['left_arm'] = LeftArm
    body_segm['right_arm'] = RightArm
    body_segm['left_leg'] = LeftLeg
    body_segm['right_leg'] = RightLeg

    df.loc[df['joint'].isin(Spine), 'part'] = 'Spine'
    df.loc[df['joint'].isin(LeftArm), 'part'] = 'LeftArm'
    df.loc[df['joint'].isin(RightArm), 'part'] = 'RightArm'
    df.loc[df['joint'].isin(LeftLeg), 'part'] = 'LeftLeg'
    df.loc[df['joint'].isin(RightLeg), 'part'] = 'RightLeg'

    ##############################################################
    ##               Retargeting algorithm                      ##
    ##############################################################

    # df['src_x'] = df['src_x']/scale['src_scale'][0]
    # print(df)

    # Try to adjust destination pose
    poses = dict()
    # get_pose_quaternion(body_segm, df, poses, bpy)
    # get_pose_euler(body_segm, df, poses, bpy)   # Best solution
    get_pose_ga(body_segm, df, poses, bpy, scales)    # GA

    # Print pose parameter
    axis, angle = get_axis_angle(poses, SMPLX_JOINT_NAMES)
    # print(axis, '\n', angle)

    # Visualize
    # fig, axes = plt.subplots(3, 2)
    # sns.scatterplot(ax=axes[0,0], data=df, x='src_x', y='src_y', hue='part')
    # sns.scatterplot(ax=axes[0,1], data=df, x='dest_x', y='dest_y', hue='part')
    # sns.scatterplot(ax=axes[1,0], data=df, x='src_x', y='src_z', hue='part')
    # sns.scatterplot(ax=axes[1,1], data=df, x='dest_x', y='dest_z', hue='part')
    # sns.scatterplot(ax=axes[2,0], data=df, x='src_z', y='src_y', hue='part')
    # sns.scatterplot(ax=axes[2,1], data=df, x='dest_z', y='dest_y', hue='part')
    # plt.show()

    # Export pkl
    result = {}
    result['bone_name'] = list(JOINTS)
    result['result'] = 'Retargeting'
    result['axis_bone'] = axis
    result['angle_bone'] = angle
    with open(out_path+'/pose_retarg.pkl', 'wb') as file:
        pickle.dump(result, file)
    
    # Export FBX
    # bpy.ops.export_scene.fbx(filepath=out_path+'retar.fbx', use_selection=False)

def get_axis_angle(poses, joints):
    axis = []; angle = []
    for p in joints[1:]:
        if p in poses.keys():
            a, n = poses[p].to_axis_angle()
            axis.append(a[:])
            angle.append(n)
        else:
            axis.append((0, 0, 0))
            angle.append(0)
    return axis, angle