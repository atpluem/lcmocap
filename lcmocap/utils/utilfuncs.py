import numpy as np
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
    
    axis = src_axis - dest_axis
    angle = angle_between(src_axis, dest_axis, '3d')
    return axis, angle