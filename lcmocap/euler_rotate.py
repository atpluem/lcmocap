from utils.utilfuncs import *

def get_pose_euler(body_segm, df, poses, bpy):
    euler_rotation(body_segm['spine'], body_segm['spine']+body_segm['left_arm']+body_segm['right_arm'], df, poses, bpy)
    euler_rotation(body_segm['left_arm'], body_segm['left_arm'], df, poses, bpy)
    euler_rotation(body_segm['right_arm'], body_segm['right_arm'], df, poses, bpy)
    euler_rotation(body_segm['left_leg'], body_segm['left_leg'], df, poses, bpy)
    euler_rotation(body_segm['right_leg'], body_segm['right_leg'], df, poses, bpy)

def euler_rotation(body_parts, update_parts, df, poses, bpy):
    Root = ['pelvis', 'spine1', 'left_hip', 'right_hip', 'left_collar', 'right_collar']
    loss_plt = {}
    for part in body_parts:
        lr = 0.072 # best lr is 0.072
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

def set_pose_euler(armature, bone_name, angle):
    if armature.pose.bones[bone_name].rotation_mode != 'YZX':
        armature.pose.bones[bone_name].rotation_mode = 'YZX'

    armature.pose.bones[bone_name].rotation_euler = angle
    # armature.pose.bones[bone_name].rotation_mode = 'QUATERNION'