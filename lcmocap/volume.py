import numpy as np
import json

def getVolumes(config, vertices, faces):

    ''' SMPL body part segmentation (24 parts 7390 unit)
        -----
        leftHand        324 rightHand       324
        leftUpLeg       254 rightUpLeg      254 // 
        leftArm         284 rightArm        284
        leftLeg         217 rightLeg        217 //
        leftToeBase     122 rightToeBase    122 
        leftFoot        143 rightFoot       143
        leftShoulder    151 rightShoulder   151
        leftHandIndex1  478 rightHandIndex1 478
        leftForeArm     246 rightForeArm    246 //
        spine           233     //
        spine1          267     //
        spine2          615     //
        head            1194
        neck            156
        hips            487     //

        Paper 17 parts
        -----
        head + neck     Hand + HandIndex1
        Shoulder + Arm  Foot + ToeBase
    '''

    # Read JSON segmentation file
    segm_path = config.body_part_segm.smpl_path
    with open(segm_path) as json_file:
        smpl_segm = json.load(json_file)

    # Union body parts (from 24 to 17)
    head_segm = list(set.union(set(smpl_segm['head']), set(smpl_segm['neck'])))
    leftHand_segm = list(set.union(set(smpl_segm['leftHand']), set(smpl_segm['leftHandIndex1'])))
    rightHand_segm = list(set.union(set(smpl_segm['rightHand']), set(smpl_segm['rightHandIndex1'])))
    leftArm_segm = list(set.union(set(smpl_segm['leftShoulder']), set(smpl_segm['leftArm'])))
    rightArm_segm = list(set.union(set(smpl_segm['rightShoulder']), set(smpl_segm['rightArm'])))
    leftFoot_segm = list(set.union(set(smpl_segm['leftFoot']), set(smpl_segm['leftToeBase'])))
    rightFoot_segm = list(set.union(set(smpl_segm['rightFoot']), set(smpl_segm['rightToeBase'])))
    spineUp_segm = list(set.union(set(smpl_segm['spine1']), set(smpl_segm['spine2'])))

    # Create seams and get coordinates
    neck_seam_coor = getSeamCoor(vertices, faces, list(set(head_segm) & set(smpl_segm['spine2'])))
    leftArm_seam_coor = getSeamCoor(vertices, faces, list(set(leftArm_segm) & set(smpl_segm['leftForeArm'])))
    rightArm_seam_coor = getSeamCoor(vertices, faces, list(set(rightArm_segm) & set(smpl_segm['rightForeArm'])))
    leftHand_seam_coor = getSeamCoor(vertices, faces, list(set(leftHand_segm) & set(smpl_segm['leftForeArm'])))
    rightHand_seam_coor = getSeamCoor(vertices, faces, list(set(rightHand_segm) & set(smpl_segm['rightForeArm'])))
    leftShoulder_seam_coor = getSeamCoor(vertices, faces, list(set(leftArm_segm) & set(smpl_segm['spine2'])))
    rightShoulder_seam_coor = getSeamCoor(vertices, faces, list(set(rightArm_segm) & set(smpl_segm['spine2'])))
    hip_seam_coor = getSeamCoor(vertices, faces, list(set(smpl_segm['spine']) & set(smpl_segm['hips'])))
    midSpine_seam_coor = getSeamCoor(vertices, faces, list(set(smpl_segm['spine']) & set(smpl_segm['spine1'])))
    leftUpLeg_seam_coor = getSeamCoor(vertices, faces, list(set(smpl_segm['leftUpLeg']) & set(smpl_segm['hips'])))
    rightUpLeg_seam_coor = getSeamCoor(vertices, faces, list(set(smpl_segm['rightUpLeg']) & set(smpl_segm['hips'])))
    leftLeg_seam_coor = getSeamCoor(vertices, faces, list(set(smpl_segm['leftLeg']) & set(smpl_segm['leftUpLeg'])))
    rightLeg_seam_coor = getSeamCoor(vertices, faces, list(set(smpl_segm['rightLeg']) & set(smpl_segm['rightUpLeg'])))
    leftFoot_seam_coor = getSeamCoor(vertices, faces, list(set(leftFoot_segm) & set(smpl_segm['leftLeg'])))
    rightFoot_seam_coor = getSeamCoor(vertices, faces, list(set(rightFoot_segm) & set(smpl_segm['rightLeg'])))

    # Get body part coordinate
    head_coor = getBPartTriCoor(vertices, faces, head_segm)
    spineUp_coor = getBPartTriCoor(vertices, faces, spineUp_segm)
    spine_coor = getBPartTriCoor(vertices, faces, smpl_segm['spine'])
    hip_coor = getBPartTriCoor(vertices, faces, smpl_segm['hips'])
    leftArm_coor = getBPartTriCoor(vertices, faces, leftArm_segm)
    rightArm_coor = getBPartTriCoor(vertices, faces, rightArm_segm)
    leftForeArm_coor = getBPartTriCoor(vertices, faces, smpl_segm['leftForeArm'])
    rightForeArm_coor = getBPartTriCoor(vertices, faces, smpl_segm['rightForeArm'])
    leftHand_coor = getBPartTriCoor(vertices, faces, leftHand_segm)
    rightHand_coor = getBPartTriCoor(vertices, faces, rightHand_segm)
    leftUpLeg_coor = getBPartTriCoor(vertices, faces, smpl_segm['leftUpLeg'])
    rightUpLeg_coor = getBPartTriCoor(vertices, faces, smpl_segm['rightUpLeg'])
    leftLeg_coor = getBPartTriCoor(vertices, faces, smpl_segm['leftLeg'])
    rightLeg_coor = getBPartTriCoor(vertices, faces, smpl_segm['rightLeg'])
    leftFoot_coor = getBPartTriCoor(vertices, faces, leftFoot_segm)
    rightFoot_coor = getBPartTriCoor(vertices, faces, rightFoot_segm)
    

    # Calculate volume    
    head_vol = getPartVolume(head_coor) + getPartVolume(neck_seam_coor)
    spineUp_vol = getPartVolume(spineUp_coor) + getPartVolume(neck_seam_coor) \
                + getPartVolume(leftArm_seam_coor) + getPartVolume(rightArm_seam_coor) \
                + getPartVolume(midSpine_seam_coor)
    spine_vol = getPartVolume(spine_coor) + getPartVolume(midSpine_seam_coor) \
                + getPartVolume(hip_seam_coor)
    hip_vol = getPartVolume(hip_coor) + getPartVolume(hip_seam_coor) \
                + getPartVolume(leftUpLeg_seam_coor) + getPartVolume(rightUpLeg_seam_coor)
    leftUpLeg_vol = getPartVolume(leftUpLeg_coor) + getPartVolume(leftUpLeg_seam_coor) \
                    + getPartVolume(leftLeg_seam_coor)
    rightUpLeg_vol = getPartVolume(rightUpLeg_coor) + getPartVolume(rightUpLeg_seam_coor) \
                    + getPartVolume(rightLeg_seam_coor)
    leftLeg_vol = getPartVolume(leftLeg_coor) + getPartVolume(leftLeg_seam_coor) \
                    + getPartVolume(leftFoot_seam_coor)
    rightLeg_vol = getPartVolume(rightLeg_coor) + getPartVolume(rightLeg_seam_coor) \
                    + getPartVolume(rightFoot_seam_coor)                
    leftFoot_vol = getPartVolume(leftFoot_coor) + getPartVolume(leftFoot_seam_coor)
    rightFoot_vol = getPartVolume(rightFoot_coor) + getPartVolume(rightFoot_seam_coor)
    leftArm_vol = getPartVolume(leftArm_coor) + getPartVolume(leftShoulder_seam_coor) \
                    + getPartVolume(leftArm_seam_coor)
    rightArm_vol = getPartVolume(rightArm_coor) + getPartVolume(rightShoulder_seam_coor) \
                    + getPartVolume(rightArm_seam_coor)
    leftForeArm_vol = getPartVolume(leftForeArm_coor) + getPartVolume(leftArm_seam_coor) \
                    + getPartVolume(leftHand_seam_coor)
    rightForeArm_vol = getPartVolume(rightForeArm_coor) + getPartVolume(rightArm_seam_coor) \
                    + getPartVolume(rightHand_seam_coor)
    leftHand_vol = getPartVolume(leftHand_coor) + getPartVolume(leftHand_seam_coor)
    rightHand_vol = getPartVolume(rightHand_coor) + getPartVolume(rightHand_seam_coor)                 

    return {'head_vol': head_vol, 'spineUp_vol': spineUp_vol, 'spine_vol': spine_vol,
            'hip_vol': hip_vol, 'leftUpLeg_vol': leftUpLeg_vol, 'rightUpLeg_vol': rightUpLeg_vol,
            'leftLeg_vol': leftLeg_vol, 'rightLeg_vol': rightLeg_vol,
            'leftFoot_vol': leftFoot_vol, 'rightFoot_vol': rightFoot_vol,
            'leftArm_vol': leftArm_vol, 'rightArm_vol': rightArm_vol,
            'leftForeArm_vol': leftForeArm_vol, 'rightForeArm_vol': rightForeArm_vol,
            'leftHand_vol': leftHand_vol, 'rightHand_vol': rightHand_vol}

def getSeamCoor(vertices, faces, vseam):

    # coordinate of centroid
    centroid = np.zeros(3)
    triangles = []

    # Find centroid and neighbors between each vertices
    for count, vertex in enumerate(vseam):
        centroid[0] += vertices[vertex][0]
        centroid[1] += vertices[vertex][1]
        centroid[2] += vertices[vertex][2]

        neighbors = set()
        # Find neighbors in all triangle face
        for triangle in faces:
            if vertex in triangle and len(set(vseam) & set(triangle)) == 2:
                neighbors.update(set(vseam) & set(triangle))

        # Create triangle of seam 
        for count, v in enumerate(neighbors):
            tri_seam = []
            if v != vertex:
                tri_seam.append(vertex)
                tri_seam.append(v)
                tri_seam.append(-1)
                triangles.append(tri_seam)
    
    tri_list = list(set([tuple(sorted(x)) for x in triangles]))
    centroid /= len(vseam)

    # Get coordinate from vertice id
    coor = []
    for triangle in tri_list:
        point = []
        point.append(centroid)
        point.append(vertices[triangle[1]])
        point.append(vertices[triangle[2]])
        coor.append(np.array(point))

    return np.array(coor)

def getBPartTriCoor(vertices, faces, vertex_id):

    coors = []
    tri_list = []
    for triangle in faces:
        if len(set(triangle) & set(vertex_id)) == 3:
            tri_list.append(triangle)
    
    for tri in tri_list:
        point = []
        point.append(vertices[tri[0]])
        point.append(vertices[tri[1]])
        point.append(vertices[tri[2]])
        coors.append(np.array(point))

    return np.array(coors)
        
def getSignedVolume(pi, pj, pk):
    vkji = pk[0]*pj[1]*pi[2]
    vjki = pj[0]*pk[1]*pi[2]
    vkij = pk[0]*pi[1]*pj[2]
    vikj = pi[0]*pk[1]*pj[2]
    vjik = pj[0]*pi[1]*pk[2]
    vijk = pi[0]*pj[1]*pk[2]
    return (1.0/6.0) * (-vkji+vjki+vkij-vikj-vjik+vijk)

def getPartVolume(triangle_coor):
    vol = 0
    for tri in triangle_coor:
        vol += getSignedVolume(tri[0], tri[1], tri[2])
    return vol