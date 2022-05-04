import time
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils.utilfuncs import *

def get_pose_euler(body_segm, df, poses, bpy, visualize):
    total_loss = dict()
    
    # initialize visualize
    sc = 0
    if visualize:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        sc = ax.scatter3D(df['dest_x'], df['dest_y'], df['dest_z'])
        fig.show()
    
    # start retargeting
    start = time.time()
    for body_set in tqdm(body_segm, desc='Retargeting'):
        if body_set == 'spine':
            update_part = body_segm[body_set] + body_segm['left_arm'] + \
                            body_segm['right_arm']
        else: update_part = body_segm[body_set]
        euler_rotation(body_set, body_segm[body_set], update_part, df, 
                        poses, bpy, total_loss, sc, visualize)
    elapsed = time.time() - start
    tqdm.write('Retargeting done after {:.4f} seconds'.format(elapsed))
    tqdm.write('Retargeting final loss val = {:.4f}'.format(get_sum_total_loss(total_loss)))
    return total_loss

def euler_rotation(body_set, body_parts, update_parts, df, poses, bpy, total_loss, sc, visualize):
    Root = ['pelvis', 'spine1', 'left_hip', 'right_hip', 'left_collar', 'right_collar']
    
    for part in tqdm(body_parts, desc='Stage {:10s}'.format(body_set)):
        stage_start = time.time()
        lr = 0.072 # best lr is 0.072
        state = 0
        pose = [0,0,0]
        min_loss = [10,10,10]
        direct = 1
        axis = [0,0,0] # [xy-pose-idx, xz-pose-idx, zy-pose-idx]
        loss_list = []
        if part in Root:
            continue

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
            loss_list.append(sum(loss))

            if visualize:
                plt.pause(0.0001)
                sc._offsets3d = (df['dest_x'], df['dest_y'], df['dest_z'])
                plt.draw()
            
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

            if state == 3:
                if parent_df['joint'].values == 'spine3':
                    if df.loc[df['joint'] == 'left_collar']['dest_x'].values < \
                       df.loc[df['joint'] == 'right_collar']['dest_x'].values:
                       pose[0] = -1 * pose[0]
                       state = 0
                       min_loss = [10,10,10]
                    else:
                        break
                elif (loss[0] > 0.35) or (loss[1] > 0.35) or (loss[2] > 0.35):
                    state = 0
                    min_loss = [10,10,10]
                else:
                    break

            set_pose_euler(bpy.data.objects['DEST'], body_parts[body_parts.index(part)-1], 
                            (pose[0], pose[1], pose[2])) # [xz, xy, zy]
            bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
            
            # Get position of each part of body part
            for child in update_parts:              
                df.loc[df['joint'] == child, ['dest_x', 'dest_y', 'dest_z']] = \
                    np.array(bpy.data.objects['DEST'].pose.bones[child].head)
        # Get result and loss
        poses[body_parts[body_parts.index(part)-1]] = \
            bpy.data.objects['DEST'].pose.bones[body_parts[body_parts.index(part)-1]].rotation_euler.to_quaternion()
        total_loss[part] = loss_list
    
        # print stage and time usage
        elapsed = time.time() - stage_start
        tqdm.write('--> {:10s} done after {:.4f} seconds'.format(part, elapsed))