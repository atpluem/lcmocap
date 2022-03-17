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

    # Scale source armature,then initial pose of both armatures 
    scale = dest_joints['head'][1] / src_joints['head'][1]
    poses = {}
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
    ##         Create dataframe for joint location              ##
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

    df.loc[df['joint'].isin(Spine), 'part'] = 'Spine'
    df.loc[df['joint'].isin(LeftArm), 'part'] = 'LeftArm'
    df.loc[df['joint'].isin(RightArm), 'part'] = 'RightArm'
    df.loc[df['joint'].isin(LeftLeg), 'part'] = 'LeftLeg'
    df.loc[df['joint'].isin(RightLeg), 'part'] = 'RightLeg'

    ##############################################################
    ##                   Calculate error                        ##
    ##############################################################

    # Try to adjust destination pose
    update_spine = Spine + LeftArm + RightArm
    # get_pose_params(Spine, update_spine, df, poses)
    # get_pose_params(LeftArm, LeftArm, df, poses)
    # get_pose_params(RightArm, RightArm, df, poses)
    # get_pose_params(LeftLeg, LeftLeg, df, poses)
    # get_pose_params(RightLeg, RightLeg, df, poses)

    # Try to adjust destination pose
    # get_pose_quaternion(Spine, update_spine, df, poses)
    # get_pose_quaternion(LeftArm, LeftArm, df, poses)
    # get_pose_quaternion(RightArm, RightArm, df, poses)
    # get_pose_quaternion(LeftLeg, LeftLeg, df, poses)
    # get_pose_quaternion(RightLeg, RightLeg, df, poses)

    # Try to adjust destination pose PASS!!!
    # get_pose_euler(Spine, update_spine, df, poses)
    # get_pose_euler(LeftArm, LeftArm, df, poses)
    # get_pose_euler(RightArm, RightArm, df, poses)
    # get_pose_euler(LeftLeg, LeftLeg, df, poses)
    # get_pose_euler(RightLeg, RightLeg, df, poses)

    # Try to adjust destination pose GA
    get_pose_genetic(Spine, update_spine, df)
    get_pose_genetic(LeftArm, LeftArm, df)
    # get_pose_genetic(RightArm, RightArm, df)
    # get_pose_genetic(LeftLeg, LeftLeg, df)
    # get_pose_genetic(RightLeg, RightLeg, df)
    
    # lp = np.array(loss_plt['spine2'])
    # print(lp)
    # plt.title('Loss of Spine2 each iterator')
    # plt.plot(lp[:-1,0], label='xy-plane')
    # plt.plot(lp[:-1,1], label='xz-plane')
    # plt.plot(lp[:-1,2], label='zy-plane')
    # plt.xlabel('number of iteration')
    # plt.ylabel('loss (radian)')
    # plt.legend()
    # plt.show()

    # Print pose parameter
    # print_pose_params(poses, SMPLX_JOINT_NAMES)

    # Print pose quaternion
    # print_pose_quat(poses, SMPLX_JOINT_NAMES)

    # Visualize
    # fig, axes = plt.subplots(2, 3)
    # sns.scatterplot(ax=axes[0,0], data=df, x='src_x', y='src_y', hue='part')
    # sns.scatterplot(ax=axes[0,1], data=df, x='src_z', y='src_y', hue='part')
    # sns.scatterplot(ax=axes[0,2], data=df, x='src_x', y='src_z', hue='part')
    # sns.scatterplot(ax=axes[1,0], data=df, x='dest_x', y='dest_y', hue='part')
    # sns.scatterplot(ax=axes[1,1], data=df, x='dest_z', y='dest_y', hue='part')
    # sns.scatterplot(ax=axes[1,2], data=df, x='dest_x', y='dest_z', hue='part')
    # plt.show()

    fig, axes = plt.subplots(3, 2)
    sns.scatterplot(ax=axes[0,0], data=df, x='src_x', y='src_y', hue='part')
    sns.scatterplot(ax=axes[0,1], data=df, x='dest_x', y='dest_y', hue='part')
    sns.scatterplot(ax=axes[1,0], data=df, x='src_x', y='src_z', hue='part')
    sns.scatterplot(ax=axes[1,1], data=df, x='dest_x', y='dest_z', hue='part')
    sns.scatterplot(ax=axes[2,0], data=df, x='src_z', y='src_y', hue='part')
    sns.scatterplot(ax=axes[2,1], data=df, x='dest_z', y='dest_y', hue='part')
    plt.show()

    # Export pkl
    # result = {}
    # result["body_pose"] = body_pose
    # print(out_path)
    # output = open(out_path+'/retar.pkl', 'wb')
    # pickle.dump(result, output)
    # output.close()

    # Export FBX
    # bpy.ops.export_scene.fbx(filepath=out_path+'retar.fbx', use_selection=False)

def set_pose_euler(armature, bone_name, angle):
    if armature.pose.bones[bone_name].rotation_mode != 'YZX':
        armature.pose.bones[bone_name].rotation_mode = 'YZX'

    armature.pose.bones[bone_name].rotation_euler = angle
    # armature.pose.bones[bone_name].rotation_mode = 'QUATERNION'
    

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

def get_pose_params(body_parts, update_parts, df, poses):
    ''' Retarget using adjusting pose parameter '''
    Root = ['pelvis', 'spine1', 'left_hip', 'right_hip', 'left_collar', 'right_collar']

    for part in body_parts:
        lr = 0.01
        min_loss = [1000,1000,1000]
        state = 0
        direct = 1
        pose = [0,0,0]
        axis = [0,0,0] # [xy-pose-idx, xz-pose-idx, zy-pose-idx]
        if part in Root:
            continue

        part_df = df.loc[df['joint'] == part]
        parent_df = df.loc[df['joint'] == body_parts[body_parts.index(part)-1]]        
        if parent_df['dest_orien'].values == 'h':
            axis = [1,0,2]
        elif parent_df['dest_orien'].values == 'v':
            axis = [2,1,0]

        print(body_parts[body_parts.index(part)-1])
        while True:    
            # Get position of each part of body part
            for child in update_parts:              
                df.loc[df['joint'] == child, ['dest_x', 'dest_y', 'dest_z']] = \
                    np.array(bpy.data.objects['DEST'].pose.bones[child].head)
            part_df = df.loc[df['joint'] == part]
            parent_df = df.loc[df['joint'] == body_parts[body_parts.index(part)-1]]

            src_angle = get_2D_angle(part_df, parent_df, 'src')
            dest_angle = get_2D_angle(part_df, parent_df, 'dest')
            loss = abs(src_angle - dest_angle) # [xy-front, xz-bottom, zy-side-right-hand]

            # if part == 'neck':
            #     print('direct: ',direct, 'axis: ', axis, 'pose: ', pose)
            #     print(state, 'angle', src_angle, dest_angle, loss, min_loss)

            if state == 0: # rotate x-axis
                if loss[2] < min_loss[2]:
                    min_loss = loss
                    pose[axis[2]] = pose[axis[2]] + direct*lr
                elif min_loss[2] > 0.05:
                    direct = -1
                    pose[axis[2]] = pose[axis[2]] - 2*lr
                else:
                    direct = 1
                    state = state + 1

            elif state == 1: # rotate y-axis
                if loss[1] < min_loss[1]:
                    min_loss = loss
                    pose[axis[1]] = pose[axis[1]] + direct*lr
                elif min_loss[1] > 0.05:
                    direct = -1
                    pose[axis[1]] = pose[axis[1]] - 2*lr
                else:
                    direct = 1
                    state = state + 1

            elif state == 2: # rotate z-axis
                if loss[0] < min_loss[0]:
                    min_loss = loss
                    pose[axis[0]] = pose[axis[0]] + direct*lr
                elif min_loss[0] > 0.05:
                    direct = -1
                    pose[axis[0]] = pose[axis[0]] - 2*lr
                else:
                    direct = 1
                    state = state + 1

            if state == 3:
                src_angle = get_3D_angle(part_df, parent_df, 'src')
                dest_angle = get_3D_angle(part_df, parent_df, 'dest')
                loss3D = abs(src_angle - dest_angle)
                print('loss3D', src_angle, dest_angle, loss3D)
                break

            # set the pose according to pose parameter
            set_pose(bpy.data.objects['DEST'], body_parts[body_parts.index(part)-1], pose)
            bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
                    
        poses[body_parts[body_parts.index(part)-1]] = pose

def get_pose_quaternion(body_parts, update_parts, df, poses):
    ''' Retarget by "axis and angle" between source and target vector '''
    Root = ['pelvis', 'spine1', 'left_hip', 'right_hip', 'left_collar', 'right_collar']

    for part in body_parts:
        min_loss3D = 0
        flag = 0
        if part in Root:
            continue

        part_df = df.loc[df['joint'] == part]
        parent_df = df.loc[df['joint'] == body_parts[body_parts.index(part)-1]] 
        print(body_parts[body_parts.index(part)-1])
        diff_axis, diff_angle = get_3D_angle_axis(part_df, parent_df)
        while True:
            q = Quaternion(-diff_axis, diff_angle)

            bpy.data.objects['DEST'].pose.bones[body_parts[body_parts.index(part)-1]].rotation_quaternion = q
            bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        
            for child in update_parts:
                df.loc[df['joint'] == child, ['dest_x', 'dest_y', 'dest_z']] = \
                    np.array(bpy.data.objects['DEST'].pose.bones[child].head)
            part_df = df.loc[df['joint'] == part]
            parent_df = df.loc[df['joint'] == body_parts[body_parts.index(part)-1]]

            if flag: break

            src_angle = get_2D_angle(part_df, parent_df, 'src')
            dest_angle = get_2D_angle(part_df, parent_df, 'dest')
            loss = abs(src_angle - dest_angle) # [xy-front, xz-bottom, zy-side-right-hand]

            src_angle = get_3D_angle(part_df, parent_df, 'src')
            dest_angle = get_3D_angle(part_df, parent_df, 'dest')
            loss3D = abs(src_angle - dest_angle)

            print(loss, src_angle, dest_angle, loss3D)
            if loss3D < min_loss3D:
                min_loss3D = loss3D
                diff_angle = diff_angle + 0.01
            else:
                # print(loss, min_loss3D)
                diff_angle = diff_angle - 0.01
                flag = 1
        poses[body_parts[body_parts.index(part)-1]] = \
            bpy.data.objects['DEST'].pose.bones[body_parts[body_parts.index(part)-1]].rotation_quaternion

def get_pose_euler(body_parts, update_parts, df, poses):
    Root = ['pelvis', 'spine1', 'left_hip', 'right_hip', 'left_collar', 'right_collar']
    loss_plt = {}
    for part in body_parts:
        lr = 0.045 # best lr is 0.072
        state = 0
        pose = [0,0,0]
        min_loss = [10,10,10]
        direct = 1
        axis = [0,0,0] # [xy-pose-idx, xz-pose-idx, zy-pose-idx]
        if part in Root:
            continue

        print(body_parts[body_parts.index(part)-1])
        part_df = df.loc[df['joint'] == part]
        parent_df = df.loc[df['joint'] == body_parts[body_parts.index(part)-1]]
        if parent_df['dest_orien'].values == 'h':
            axis = [2,0,1]
        elif parent_df['dest_orien'].values == 'v':
            poses[body_parts[body_parts.index(part)-1]] = \
                bpy.data.objects['SRC'].pose.bones[body_parts[body_parts.index(part)-1]].rotation_quaternion
            continue
        
        while True:
            part_df = df.loc[df['joint'] == part]
            parent_df = df.loc[df['joint'] == body_parts[body_parts.index(part)-1]]

            src_angle = get_2D_angle(part_df, parent_df, 'src')
            dest_angle = get_2D_angle(part_df, parent_df, 'dest')
            loss = abs(src_angle - dest_angle) # [xy-front, xz-bottom, zy-side-right-hand]     

            if state == 0: # rotate x-axis
                dloss = abs(loss[2] - min_loss[2])
                if (loss[2] < min_loss[2]) and (dloss > 0.001):
                    min_loss[2] = loss[2]
                    pose[axis[0]] = pose[axis[0]] + direct*lr
                elif (min_loss[2] > 0.05) and (dloss > 0.001):
                    direct = -1
                    pose[axis[0]] = pose[axis[0]] - 2*lr
                elif (min_loss[2] < 0.05) or (dloss < 0.001):
                    pose[axis[0]] = pose[axis[0]] + lr
                    direct = 1
                    state = state + 1

            elif state == 1: # rotate y-axis
                dloss = abs(loss[1] - min_loss[1])
                if (loss[1] < min_loss[1]) and (dloss > 0.001):
                    min_loss[1] = loss[1]
                    pose[axis[1]] = pose[axis[1]] + direct*lr
                elif (min_loss[1] > 0.05) and (dloss > 0.001):
                    direct = -1
                    pose[axis[1]] = pose[axis[1]] - 2*lr
                elif (min_loss[1] < 0.05) or (dloss < 0.001):
                    pose[axis[1]] = pose[axis[1]] + lr
                    direct = 1
                    state = state + 1

            elif state == 2: # rotate z-axis
                dloss = abs(loss[0] - min_loss[0])
                if (loss[0] < min_loss[0]) and (dloss > 0.001):
                    min_loss[0] = loss[0]
                    pose[axis[2]] = pose[axis[2]] + direct*lr
                elif (min_loss[0] > 0.05) and (dloss > 0.001):
                    direct = -1
                    pose[axis[2]] = pose[axis[2]] - 2*lr
                elif (min_loss[0] < 0.05) or (dloss < 0.001):
                    pose[axis[2]] = pose[axis[2]] + lr
                    direct = 1
                    state = state + 1

            # if body_parts[body_parts.index(part)-1] == 'spine1':
            #     print('state: ', state, pose)
            #     print('loss', src_angle, dest_angle, loss, min_loss)

            if state == 3:
                if parent_df['joint'].values == 'spine3':
                    if df.loc[df['joint'] == 'left_collar']['dest_x'].values < \
                       df.loc[df['joint'] == 'right_collar']['dest_x'].values:
                       pose[0] = -1 * pose[0]
                       state = 0
                       min_loss = [10,10,10]
                    else:
                        print('loss: ', loss)
                        break
                elif (loss[0] > 0.35) or (loss[1] > 0.35) or (loss[2] > 0.35):
                    # print('try')
                    state = 0
                    min_loss = [10,10,10]
                else:
                    print('loss: ', loss)
                    break

            set_pose_euler(bpy.data.objects['DEST'], body_parts[body_parts.index(part)-1], 
                            (pose[0], pose[1], pose[2])) # [xz, xy, zy]
            bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
            
            # Get position of each part of body part
            for child in update_parts:              
                df.loc[df['joint'] == child, ['dest_x', 'dest_y', 'dest_z']] = \
                    np.array(bpy.data.objects['DEST'].pose.bones[child].head)
            
        poses[body_parts[body_parts.index(part)-1]] = \
            bpy.data.objects['DEST'].pose.bones[body_parts[body_parts.index(part)-1]].rotation_euler.to_quaternion()

def get_pose_genetic(body_parts, update_parts, df):
    Root = ['pelvis', 'spine1', 'left_hip', 'right_hip', 
            'left_collar', 'right_collar']
    poses = list()
    for part in body_parts:
        if part in Root: continue
        print(body_parts[body_parts.index(part)-1])
        
        pose_bound = [[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0]]
        n_iter = 30
        n_bits = 32
        n_pop = 100
        r_cross = 0.9
        r_mut = 1.0 / (float(n_bits) * len(pose_bound))
        parent = body_parts[body_parts.index(part)-1]
        child = part

        # initial population of random bitstring
        pop = [randint(0, 2, n_bits*len(pose_bound)).tolist() for _ in range(n_pop)]
        # track of best solution
        best, best_eval = 0, objective_loss(decode(pose_bound, n_bits, pop[0]),
                                              child, parent, update_parts, df)
        
        # enumerate generations
        for gen in range(n_iter):
            # decode population
            decoded = [decode(pose_bound, n_bits, p) for p in pop]
            # evaluate all candidates in the population
            scores = [objective_loss(d, child, parent, update_parts, df) \
                                    for d in decoded]
            # check for new best solution
            for i in range(n_pop):
                if scores[i] < best_eval:
                    best, best_eval = pop[i], scores[i]
                    print(">%d, new best f(%s) = %f" %(gen,  decoded[i], scores[i]))
            # select parents
            selected = [selection(pop, scores) for _ in range(n_pop)]
            # create the next generation
            children = list()
            for i in range(0, n_pop, 2):
                # selected parents in pairs
                p1, p2 = selected[i], selected[i+1]
                # crossover and mutation
                for c in crossover(p1, p2, r_cross):
                    mutation(c, r_mut)
                    children.append(c)
            pop = children
        
        pose = decode(pose_bound, n_bits, best)
        print('pose: %s, loss: %f'%(pose, best_eval))
        # poses[body_parts[body_parts.index(part)-1]] = pose

def objective_loss(pose, child, parent, update_parts, df):
    # set the pose according to pose parameter
    set_pose(bpy.data.objects['DEST'], parent, pose)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

    # Get position of each part of body part
    for part in update_parts:              
        df.loc[df['joint'] == part, ['dest_x', 'dest_y', 'dest_z']] = \
            np.array(bpy.data.objects['DEST'].pose.bones[part].head)
    part_df = df.loc[df['joint'] == child]
    parent_df = df.loc[df['joint'] == parent]

    src_angle = get_2D_angle(part_df, parent_df, 'src')
    dest_angle = get_2D_angle(part_df, parent_df, 'dest')
    loss = abs(src_angle - dest_angle)
    return sum(loss)

def mutation(bitstring, r_mut):
	for i in range(len(bitstring)):
		# check for a mutation
		if rand() < r_mut:
			# flip the bit
			bitstring[i] = 1 - bitstring[i]

def crossover(p1, p2, r_cross):
	# children are copies of parents by default
	c1, c2 = p1.copy(), p2.copy()
	# check for recombination
	if rand() < r_cross:
		# select crossover point that is not on the end of the string
		pt = randint(1, len(p1)-2)
		# perform crossover
		c1 = p1[:pt] + p2[pt:]
		c2 = p2[:pt] + p1[pt:]
	return [c1, c2]

def selection(pop, scores, k=3):
	# first random selection
	selection_ix = randint(len(pop))
	for ix in randint(0, len(pop), k-1):
		# check if better (e.g. perform a tournament)
		if scores[ix] < scores[selection_ix]:
			selection_ix = ix
	return pop[selection_ix]

def decode(bounds, n_bits, bitstring):
	decoded = list()
	largest = 2**n_bits
	for i in range(len(bounds)):
		# extract the substring
		start, end = i * n_bits, (i * n_bits)+n_bits
		substring = bitstring[start:end]
		# convert bitstring to a string of chars
		chars = ''.join([str(s) for s in substring])
		# convert string to integer
		integer = int(chars, 2)
		# scale integer to desired range
		value = bounds[i][0] + (integer/largest) * (bounds[i][1] - bounds[i][0])
		decoded.append(value)
	return decoded

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2, dim='2d'):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)

    if dim == '2d':
        return np.arccos(np.dot(v1_u, v2_u))
    elif dim == '3d':
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def get_2D_angle(part_df, parent_df, targ):
    dx = part_df[targ+'_x'].values - parent_df[targ+'_x'].values
    dy = part_df[targ+'_y'].values - parent_df[targ+'_y'].values
    dz = part_df[targ+'_z'].values - parent_df[targ+'_z'].values
    
    return np.array([angle_between([dx[0], dy[0]], [1,0]),
                     angle_between([dx[0], dz[0]], [1,0]),
                     angle_between([dz[0], dy[0]], [1,0])])

def get_3D_angle(part_df, parent_df, targ):
    dx = part_df[targ+'_x'].values - parent_df[targ+'_x'].values
    dy = part_df[targ+'_y'].values - parent_df[targ+'_y'].values
    dz = part_df[targ+'_z'].values - parent_df[targ+'_z'].values
    
    return np.array([angle_between([dx[0], dz[0], dy[0]], [1,0,0], '3d')])

def get_3D_angle_axis(part_df, parent_df):
    dx = part_df['src_x'].values - parent_df['src_x'].values
    dy = part_df['src_y'].values - parent_df['src_y'].values
    dz = part_df['src_z'].values - parent_df['src_z'].values
    src_axis = unit_vector([dx[0], dy[0], dz[0]])

    dx = part_df['dest_x'].values - parent_df['dest_x'].values
    dy = part_df['dest_y'].values - parent_df['dest_y'].values
    dz = part_df['dest_z'].values - parent_df['dest_z'].values
    dest_axis = unit_vector([dx[0], dy[0], dz[0]])
    
    axis = src_axis - dest_axis
    angle = angle_between(src_axis, dest_axis, '3d')
    return axis, angle

def mat_offset(pose_bone):
    bone = pose_bone.bone
    mat = bone.matrix.to_4x4()
    mat.translation = bone.head
    if pose_bone.parent:
        mat.translation.y += bone.parent.length
    return mat

def print_pose_quat(poses, joints):
    axi = []
    ang = []
    for i in joints[1:]:
        if i in poses.keys():
            a, n = poses[i].to_axis_angle()
            axi.append(a[:])
            ang.append(n)
        else:
            axi.append((0,0,0))
            ang.append(0)

    print('[')
    for i in axi:
        print(i,',')
    print(']')
    print('[')
    for i in ang:
        print(i,',')
    print(']')

def print_pose_params(poses, joints):
    out = []
    for i in joints[1:]:
        if i in poses.keys():
            out.append(poses[i])
        else:
            out.append([0,0,0])
    
    print('[')
    for i in out:
        print(i,',')
    print(']')