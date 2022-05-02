import time
import matplotlib.pyplot as plt

from utils.utilfuncs import *

def get_pose_glob_rotate(body_segm, df, poses, bpy, visualize=False):
    total_loss = dict()
    sc = 0
    if visualize:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        sc = ax.scatter3D(df['dest_x'], df['dest_y'], df['dest_z'])
        fig.show()
    
    # Set armature to POSE mode
    bpy.data.objects['DEST'].select_set(True)
    bpy.ops.object.mode_set(mode='POSE')

    start = time.time()
    euler_rotation(body_segm['spine'], body_segm['spine']+body_segm['left_arm']+\
                   body_segm['right_arm'], df, poses, bpy, total_loss, sc, visualize)
    euler_rotation(body_segm['left_arm'], body_segm['left_arm'], df, poses, bpy, total_loss, sc, visualize)
    euler_rotation(body_segm['right_arm'], body_segm['right_arm'], df, poses, bpy, total_loss, sc, visualize)
    euler_rotation(body_segm['left_leg'], body_segm['left_leg'], df, poses, bpy, total_loss, sc, visualize)
    euler_rotation(body_segm['right_leg'], body_segm['right_leg'], df, poses, bpy, total_loss, sc, visualize)
    stop = time.time()
    print('Take time:', stop-start)
    
    return total_loss

def euler_rotation(body_parts, update_parts, df, poses, bpy, total_loss, sc, visualize):
    Root = ['pelvis', 'spine1', 'left_hip', 'right_hip', 'left_collar', 'right_collar']

    for part in body_parts:
        state = 0
        pose = [0,0,0]
        min_loss = [10,10,10]
        direct = 1
        loss_list = []
        if part in Root:
            continue

        print(body_parts[body_parts.index(part)-1])
        
        while True:
            part_df = df.loc[df['joint'] == part]
            parent_df = df.loc[df['joint'] == body_parts[body_parts.index(part)-1]]

            src_angle = get_2D_angle(part_df, parent_df, 'src')
            dest_angle = get_2D_angle(part_df, parent_df, 'dest')
            loss = src_angle - dest_angle # [xy-front, xz-bottom, zy-side-right-hand]     
            loss_list.append(sum(abs(loss)))

            if visualize:
                plt.pause(0.001)
                sc._offsets3d = (df['dest_x'], df['dest_y'], df['dest_z'])
                plt.draw()

            # if sum(abs(loss)) < 10 :
            #     if parent_df['joint'].values == 'spine3':
            #         if df.loc[df['joint'] == 'left_collar']['dest_x'].values < \
            #            df.loc[df['joint'] == 'right_collar']['dest_x'].values:
            #            pose[0] = -1 * pose[0]
            #            state = 0
            #            min_loss = [10,10,10]
            #         else:
            #             print('loss: ', loss)
            #             break
            #     elif (loss[0] > 0.35) or (loss[1] > 0.35) or (loss[2] > 0.35):
            #         state = 0
            #         min_loss = [10,10,10]
            #     else:
            #         print('loss: ', loss)
            #         break

            set_pose_global(bpy.data.objects['DEST'], body_parts[body_parts.index(part)-1], bpy, loss)
            # Update position of each part of body part
            for child in update_parts:              
                df.loc[df['joint'] == child, ['dest_x', 'dest_y', 'dest_z']] = \
                    np.array(bpy.data.objects['DEST'].pose.bones[child].head)

            break

        poses[body_parts[body_parts.index(part)-1]] = \
            bpy.data.objects['DEST'].pose.bones[body_parts[body_parts.index(part)-1]].rotation_quaternion
        total_loss[part] = loss_list

def set_pose_global(armature, bone_name, bpy, angle):
    
    armature.data.bones[bone_name].select = True
    bpy.ops.transform.rotate(value = angle[0], orient_axis='Z', orient_type='GLOBAL')
    bpy.ops.transform.rotate(value = angle[1], orient_axis='Y', orient_type='GLOBAL')
    bpy.ops.transform.rotate(value = angle[2], orient_axis='X', orient_type='GLOBAL')
    armature.data.bones[bone_name].select = False

    print('after: ',armature.pose.bones[bone_name].rotation_quaternion)