import time
from utils.utilfuncs import *

def get_pose_quaternion(body_segm, df, poses, bpy):
    total_loss = dict()
    start = time.time()
    quaternion_rotation(body_segm['spine'], body_segm['spine']+body_segm['left_arm']+\
                        body_segm['right_arm'], df, poses, bpy, total_loss)
    # quaternion_rotation(body_segm['left_arm'], body_segm['left_arm'], df, poses, bpy, total_loss)
    # quaternion_rotation(body_segm['right_arm'], body_segm['right_arm'], df, poses, bpy, total_loss)
    # quaternion_rotation(body_segm['left_leg'], body_segm['left_leg'], df, poses, bpy, total_loss)
    # quaternion_rotation(body_segm['right_leg'], body_segm['right_leg'], df, poses, bpy, total_loss)
    stop = time.time()
    print('Take time:', stop-start)

    return total_loss

def quaternion_rotation(body_parts, update_parts, df, poses, bpy, total_loss):
    Root = ['pelvis', 'spine1', 'left_hip', 'right_hip', 'left_collar', 'right_collar']

    for part in body_parts:
        min_loss3D = 0
        flag = 0
        if part in Root: continue

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