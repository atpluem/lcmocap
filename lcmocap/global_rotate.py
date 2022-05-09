import time
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils.utilfuncs import *

def get_pose_glob_rotate(body_segm, df, poses, bpy, visualize=False):
    total_loss = dict()
    
    # initialize visualize
    sc = 0
    if visualize:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        sc = ax.scatter3D(df['dest_x'], df['dest_y'], df['dest_z'])
        fig.show()
    
    # Set armature to POSE mode
    bpy.data.objects['DEST'].select_set(True)
    bpy.ops.object.mode_set(mode='POSE')
    # DESELECT all bones
    for bone in bpy.context.selected_pose_bones_from_active_object:
        bpy.data.objects['DEST'].data.bones[bone.name].select = False

    # start retargeting
    start = time.time()
    i=0
    for body_set in tqdm(body_segm, desc='Retargeting'):
        if body_set == 'spine':
            update_part = body_segm[body_set] + body_segm['left_arm'] + \
                            body_segm['right_arm']
        else: update_part = body_segm[body_set]
        global_rotation(body_set, body_segm[body_set], update_part, df, 
                        poses, bpy, total_loss, sc, visualize)
        if i == 4: break
        i=i+1
    elapsed = time.time() - start 

    # Set armature back to OBJECT mode
    bpy.ops.object.mode_set(mode='OBJECT')
    tqdm.write('Retargeting done after {:.4f} seconds'.format(elapsed))
    tqdm.write('Retargeting final loss val = {:.4f}'.format(get_sum_total_loss(total_loss)))
    return total_loss

def global_rotation(body_set, body_parts, update_parts, df, poses, bpy, total_loss, sc, visualize):
    Root = ['pelvis', 'spine1', 'left_hip', 'right_hip', 'left_collar', 'right_collar']

    for part in tqdm(body_parts, desc='Stage {:10s}'.format(body_set)):
        stage_start = time.time()
        lr = 0.05
        pose = 0
        state = 0
        min_loss = [10,10,10]
        direction = 1
        loss_list = []

        if part in Root: continue

        while True:
            part_df = df.loc[df['joint'] == part]
            parent_df = df.loc[df['joint'] == body_parts[body_parts.index(part)-1]]

            src_angle = get_2D_angle(part_df, parent_df, 'src')
            dest_angle = get_2D_angle(part_df, parent_df, 'dest')
            loss = abs(src_angle - dest_angle) # [xy-front, xz-bottom, zy-side-right-hand]     
            loss_list.append(sum(loss))

            print(part, state, direction, loss, '=== min loss: ', min_loss)

            if visualize:
                plt.pause(0.0001)
                sc._offsets3d = (df['dest_x'], df['dest_y'], df['dest_z'])
                plt.draw()
            
            if state == 0: # rotate z-axis
                pose, state, direction = check_state(state, loss, min_loss, lr, direction)

            elif state == 1: # rotate y-axis
                pose, state, direction = check_state(state, loss, min_loss, lr, direction)

            elif state == 2: # rotate x-axis
                pose, state, direction = check_state(state, loss, min_loss, lr, direction)

            if state == 3:
                if parent_df['joint'].values == 'spine3':
                    if df.loc[df['joint'] == 'left_collar']['dest_x'].values < \
                       df.loc[df['joint'] == 'right_collar']['dest_x'].values:
                       state = 0
                       min_loss = [10,10,10]
                    else: break
                elif (loss[0] > 0.55) or (loss[1] > 0.55) or (loss[2] > 0.55):
                    state = 0
                    min_loss = [10,10,10]
                else: break

            set_pose_global(bpy.data.objects['DEST'], body_parts[body_parts.index(part)-1], bpy, state, pose)
            # Get position of each part of body part
            for child in update_parts:              
                df.loc[df['joint'] == child, ['dest_x', 'dest_y', 'dest_z']] = \
                    np.array(bpy.data.objects['DEST'].pose.bones[child].head)

        poses[body_parts[body_parts.index(part)-1]] = \
            bpy.data.objects['DEST'].pose.bones[body_parts[body_parts.index(part)-1]].rotation_quaternion
        total_loss[part] = loss_list

        # print stage and time usage
        # elapsed = time.time() - stage_start
        # tqdm.write('--> {:10s} done after {:.4f} seconds'.format(part, elapsed))

def set_pose_global(armature, bone_name, bpy, state, angle):
    armature.data.bones[bone_name].select = True
    if state == 0: bpy.ops.transform.rotate(value=angle, orient_axis='Z', orient_type='GLOBAL')
    elif state == 1: bpy.ops.transform.rotate(value=angle, orient_axis='Y', orient_type='GLOBAL')
    elif state == 2: bpy.ops.transform.rotate(value=angle, orient_axis='X', orient_type='GLOBAL')
    armature.data.bones[bone_name].select = False

def check_state(state, loss, min_loss, lr, direction):
    dloss = abs(loss[state] - min_loss[state])
    if (loss[state] < min_loss[state]) and (dloss > 0.001):
        min_loss[state] = loss[state]
        pose = direction*lr
    elif (min_loss[state] > 0.1) and (dloss > 0.001):
        direction = -direction
        pose = -lr
    elif (min_loss[state] < 0.1) or (dloss < 0.001):
        pose = lr
        direction = 1
        state = state + 1
    return pose, state, direction