import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

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

def quadrant(vec):
    if vec[0] >= 0 and vec[1] >= 0: return 1
    elif vec[0] < 0 and vec[1] >= 0: return 2
    elif vec[0] < 0 and vec[1] < 0: return 3
    elif vec[0] >= 0 and vec[1] < 0: return 4

def get_2D_quadrant(part_df, parent_df, targ):
    dx = part_df[targ+'_x'].values - parent_df[targ+'_x'].values
    dy = part_df[targ+'_y'].values - parent_df[targ+'_y'].values
    dz = part_df[targ+'_z'].values - parent_df[targ+'_z'].values

    return np.array([quadrant([dx[0], dy[0]]),
                    quadrant([dx[0], dz[0]]),
                    quadrant([dz[0], dy[0]])])

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

def joint_coordinate_plot(df):
    fig, axes = plt.subplots(3, 2)
    sns.scatterplot(ax=axes[0,0], data=df, x='src_x', y='src_y', hue='part')
    sns.scatterplot(ax=axes[0,1], data=df, x='dest_x', y='dest_y', hue='part')
    sns.scatterplot(ax=axes[1,0], data=df, x='src_x', y='src_z', hue='part')
    sns.scatterplot(ax=axes[1,1], data=df, x='dest_x', y='dest_z', hue='part')
    sns.scatterplot(ax=axes[2,0], data=df, x='src_z', y='src_y', hue='part')
    sns.scatterplot(ax=axes[2,1], data=df, x='dest_z', y='dest_y', hue='part')
    plt.show()

def convert_angle_quadrant(angle, quad):
    ang = angle
    for i, axis in enumerate(quad):
        if axis in [3, 4]:
            ang[i] = 2*math.pi - ang[i]
    return ang

def add_end_bone(bpy, armature, end_bone, addition_bone, trans):    
    select_one_object(bpy, armature)
    head_coor = armature.pose.bones[end_bone].head
    bpy.ops.object.mode_set(mode='EDIT', toggle=False)
    edit_bones = armature.data.edit_bones
    b = edit_bones.new(addition_bone)
    b.head = (head_coor[0]+trans[0], head_coor[1]+trans[1], head_coor[2]+trans[2])
    b.tail = (head_coor[0]+trans[0], head_coor[1]+trans[1]+5, head_coor[2]+trans[2])
    b.parent = armature.data.edit_bones['head']
    bpy.ops.object.mode_set(mode='OBJECT')

def select_one_object(bpy, obj):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
