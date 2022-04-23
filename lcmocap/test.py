import bpy

bpy.ops.mesh.primitive_monkey_add()
bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
