import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from mathutils import Vector, Quaternion, Matrix

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

def set_pose_euler(armature, bone_name, angle):
    if armature.pose.bones[bone_name].rotation_mode != 'YZX':
        armature.pose.bones[bone_name].rotation_mode = 'YZX'

    armature.pose.bones[bone_name].rotation_euler = angle
    # armature.pose.bones[bone_name].rotation_mode = 'QUATERNION'

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
    
    # axis = Vector(dest_axis- src_axis).normalized()
    axis = Vector(np.cross(src_axis, dest_axis)).normalized()
    angle = angle_between(src_axis, dest_axis, '3d')
    
    return axis, angle

def new_domain(src, dest, src_mesh, dest_mesh):
    src_shoulder = [src.pose.bones['left_shoulder'].head, src.pose.bones['right_shoulder'].head]
    src_hip = [src.pose.bones['left_hip'].head, src.pose.bones['right_hip'].head]
    dest_shoulder = [dest.pose.bones['left_shoulder'].head, dest.pose.bones['right_shoulder'].head]
    dest_hip = [dest.pose.bones['left_hip'].head, dest.pose.bones['right_hip'].head]
    
    src_mesh_lst = np.array([list(x.co) for x in src_mesh.data.vertices])
    dest_mesh_lst = np.array([list(x.co) for x in dest_mesh.data.vertices])

    # filter only middle body part of mesh
    src_mesh_lst = src_mesh_lst[(src_mesh_lst[:,0] > src_shoulder[1][0]) &
                                (src_mesh_lst[:,0] < src_shoulder[0][0]) &
                                (src_mesh_lst[:,1] > src_hip[0][1]) &
                                (src_mesh_lst[:,1] < src_shoulder[0][1])]
    dest_mesh_lst = dest_mesh_lst[(dest_mesh_lst[:,0] > dest_shoulder[1][0]) &
                                (dest_mesh_lst[:,0] < dest_shoulder[0][0]) &
                                (dest_mesh_lst[:,1] > dest_hip[0][1]) &
                                (dest_mesh_lst[:,1] < dest_shoulder[0][1])]

    # now we can calcualte scale only x, y axis
    src_scale = [(src_shoulder[0][0] - src_shoulder[1][0])/2,
                (src_shoulder[0][1] - src_hip[0][1])/2,
                src_mesh_lst[:,2].max()] 
    src_center = [(src_shoulder[0][0] + src_shoulder[1][0])/2, 
                (src_shoulder[0][1] + src_hip[0][1])/2]
    
    dest_scale = [(dest_shoulder[0][0] - dest_shoulder[1][0])/2,
                (dest_shoulder[0][1] - dest_hip[0][1])/2,
                dest_mesh_lst[:,2].max()] 
    dest_center = [(dest_shoulder[0][0] + dest_shoulder[1][0])/2, 
                (dest_shoulder[0][1] + dest_hip[0][1])/2]

    return {'src_scale': src_scale, 'dest_scale': dest_scale}

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

def get_sum_total_loss(total_loss):
    sum_loss = 0
    for k in total_loss.keys():
        sum_loss += total_loss[k][-1]
    return sum_loss

def total_loss_plot(total_loss, max_itr):
    losses = np.zeros((max_itr,))

    for k in total_loss.keys():
        for idx, loss in enumerate(total_loss[k]):
            if idx >= 30: break
            losses[idx] += loss
    
    sns.lineplot(x=range(max_itr), y=losses).set(title='Sum of loss each iteration',
        xlabel='number of iteration', ylabel='total loss')
    plt.show()