import os
import numpy as np
from mathutils import Vector, Quaternion
import pickle

import bpy
from bpy_extras.io_utils import ImportHelper
from bpy.props import ( BoolProperty, EnumProperty, FloatProperty, PointerProperty, StringProperty )
from bpy.types import ( PropertyGroup )

bl_info = {
    "name": "PKL Covertor mapping model for blender",
    "author": "Krittanan Chalong",
    "version": (1, 0),
    "blender": (2, 80, 0),
    "location": "Viewport > Right panel",
    "description": "PKL Covertor mapping model for blender",
    "warning": "",
    "doc_url": "",
    "category": "PKL",
}

SMPLX_JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
    'jaw', 'left_eye_smplhf', 'right_eye_smplhf', 'left_index1', 'left_index2', 'left_index3', 'left_middle1', 'left_middle2', 'left_middle3', 'left_pinky1', 'left_pinky2', 'left_pinky3', 'left_ring1', 'left_ring2', 'left_ring3', 'left_thumb1', 'left_thumb2', 'left_thumb3', 'right_index1', 'right_index2', 'right_index3', 'right_middle1', 'right_middle2', 'right_middle3', 'right_pinky1', 'right_pinky2', 'right_pinky3', 'right_ring1', 'right_ring2', 'right_ring3', 'right_thumb1', 'right_thumb2', 'right_thumb3'
]
NUM_SMPLX_JOINTS = len(SMPLX_JOINT_NAMES)
NUM_SMPLX_BODYJOINTS = 21
NUM_SMPLX_HANDJOINTS = 15

# BONE_LIST = []

def set_pose_SMPLify(armature, bone_name, rodrigues, rodrigues_reference=None):
    rod = Vector((rodrigues[0], rodrigues[1], rodrigues[2]))
    angle_rad = rod.length
    axis = rod.normalized()

    if armature.pose.bones[bone_name].rotation_mode != "QUATERNION":
        armature.pose.bones[bone_name].rotation_mode = "QUATERNION"

    quat = Quaternion(axis, angle_rad)

    if rodrigues_reference is None:
        armature.pose.bones[bone_name].rotation_quaternion = quat
    else:
        rod_reference = Vector((rodrigues[0], rodrigues[1], rodrigues[2]))
        rod_result = rod + rod_reference
        angle_rad_result = rod_result.length
        axis_result = rod_result.normalized()
        quat_result = Quaternion(axis_result, angle_rad_result)
        armature.pose.bones[bone_name].rotation_quaternion = quat_result
    return


def set_pose_Retargeting(armature, bone_name, axis, angle):

    if armature.pose.bones[bone_name].rotation_mode != "QUATERNION":
        armature.pose.bones[bone_name].rotation_mode = "QUATERNION"

    quat = Quaternion(axis, angle)
    armature.pose.bones[bone_name].rotation_quaternion = quat

    return


# def list_bone():
#     bone_list = []
#     obj = bpy.context.object

#     if obj.type == 'MESH':
#         armature = obj.parent
#     else:
#         armature = obj
#         obj = armature.children[0]

#     for bone in armature.pose.bones:
#         bone_list.append((bone.name, ""))

#     return bone_list


class PKLLoadfile_Operator(bpy.types.Operator, ImportHelper):
    bl_idname = "object.pkl_import_file"
    bl_label = "Load PKL"
    bl_description = "Load data of position body from pkl file"
    bl_options = {"REGISTER", "UNDO"}

    filter_glod: StringProperty(
        default="*.pkl",
        option={'HIDDEN'}
    )

    update_shape: BoolProperty(
        name="Update shape parameter",
        default=True
    )

    @classmethod
    def poll(cls, context):
        try:
            if (context.active_object is not None):
                return True
            else:
                return False
        except:
            return False

    def execute(self, context):
        armature = bpy.context.object

        if armature.type == "MESH":
            armature = armature.parent
        elif armature.type != "ARMATURE":
            return {"FINISHED"}

        # if armature == None:
        #     return {"FINISHED"}

        # body_pose = None

        with open(self.filepath, "rb") as f:
            data = pickle.load(f, encoding="latin1")

            if "result" in data:
                if data["result"] == "Retargeting":
                    bone_name_retarget = data["bone_name"]
                    axis_bone_retarget = data["axis_bone"]
                    angle_bone_retarget = data["angle_bone"]

                    for index in range(NUM_SMPLX_BODYJOINTS):
                        set_pose_Retargeting(armature, bone_name_retarget[index + 1], axis_bone_retarget[index], angle_bone_retarget[index])

                    # for index in range(NUM_SMPLX_BODYJOINTS):
                    #     bone_name_retarget = SMPLX_JOINT_NAMES[index + 1]
                    #     set_pose_Retargeting(armature, bone_name_retarget, axis_bone[index], angle_bone[index])

            elif "body_pose" in data:
                body_pose = np.array(data["body_pose"])
                if body_pose.shape != (1, NUM_SMPLX_BODYJOINTS * 3):
                    print(f"Invalid body pose dimensions: {body_pose.shape}")
                    body_data = None
                    return {'CANCELLED'}

                body_pose = np.array(data["body_pose"]).reshape(NUM_SMPLX_BODYJOINTS, 3)

                for index in range(NUM_SMPLX_BODYJOINTS):
                    pose_rodrigues = body_pose[index]
                    bone_name = SMPLX_JOINT_NAMES[index + 1] # body pose starts with left_hip
                    set_pose_SMPLify(armature, bone_name, pose_rodrigues)

        return {"FINISHED"}


# class List_All_Bone_Operator(bpy.types.Operator):
#     bl_idname = "object.all_bones"
#     bl_label = "List Bone"
#     bl_description = "List all bones of that model"
#     bl_options = {"REGISTER"}

#     @classmethod
#     def poll(cls, context):
#         try:
#             if (context.active_object is not None) or (context.active_object.mode == 'OBJECT'):
#                 return True
#             else: 
#                 return False
#         except: return False

#     def execute(self, context):
#         obj = bpy.context.object

#         if obj.type == 'MESH':
#             armature = obj.parent
#         else:
#             armature = obj
#             obj = armature.children[0]
#             context.view_layer.objects.active = obj

#         for bone in armature.pose.bones:
#             BONE_LIST.append((bone.name, ""))
        
#         print(BONE_LIST)
        
#         return {"FINISHED"}


class Import_PKL_Panel(bpy.types.Panel):
    bl_label = "Import File"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "PKL_Convertor"

    def draw(self, context):
        layout = self.layout

        row = layout.row()
        row.scale_y = 2.0
        row.operator("object.pkl_import_file")


# class AllBone_Panel(bpy.types.Panel):
#     bl_label = "Bone"
#     bl_space_type = "VIEW_3D"
#     bl_region_type = "UI"
#     bl_category = "PKL_Convertor"

#     def draw(self, context):
#         layout = self.layout

#         col = layout.column(align=True)
#         # row = col.row(align=True)
        
#         col.prop(context.window_manager.pkl_tools, 'pkl_bone_name')
#         col.operator("object.all_bones")


# class PKL_Properties(PropertyGroup):

#     pkl_bone_name: EnumProperty(
#         name = "Bone",
#         description = "List all bone of model",
#         items = BONE_LIST
#     )


ADDON_CLASS = [
    Import_PKL_Panel,
    PKLLoadfile_Operator,
]


def register():
    for classes in ADDON_CLASS:
        bpy.utils.register_class(classes)
    
    # bpy.types.WindowManager.pkl_tools = PointerProperty(type=PKL_Properties)


def unregister():
    for classes in ADDON_CLASS:
        bpy.utils.unregister_class(classes)

    # del bpy.types.WindowManager.pkl_tools


if __name__ == "__main__":
    register()
