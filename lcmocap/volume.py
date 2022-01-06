import numpy as np
import json

def getVolumes(vertices, faces, smpl_segm):

    # Create seams and get coordinates
    neck_seam_coor = getSeamCoor(vertices, faces, list(set(smpl_segm['head']) & set(smpl_segm['spineUp'])))
    leftArm_seam_coor = getSeamCoor(vertices, faces, list(set(smpl_segm['leftArm']) & set(smpl_segm['leftForeArm'])))
    rightArm_seam_coor = getSeamCoor(vertices, faces, list(set(smpl_segm['rightArm']) & set(smpl_segm['rightForeArm'])))
    leftHand_seam_coor = getSeamCoor(vertices, faces, list(set(smpl_segm['leftHand']) & set(smpl_segm['leftForeArm'])))
    rightHand_seam_coor = getSeamCoor(vertices, faces, list(set(smpl_segm['rightHand']) & set(smpl_segm['rightForeArm'])))
    leftShoulder_seam_coor = getSeamCoor(vertices, faces, list(set(smpl_segm['leftArm']) & set(smpl_segm['spineUp'])))
    rightShoulder_seam_coor = getSeamCoor(vertices, faces, list(set(smpl_segm['rightArm']) & set(smpl_segm['spineUp'])))
    hips_seam_coor = getSeamCoor(vertices, faces, list(set(smpl_segm['spine']) & set(smpl_segm['hips'])))
    midSpine_seam_coor = getSeamCoor(vertices, faces, list(set(smpl_segm['spine']) & set(smpl_segm['spineUp'])))
    leftUpLeg_seam_coor = getSeamCoor(vertices, faces, list(set(smpl_segm['leftUpLeg']) & set(smpl_segm['hips'])))
    rightUpLeg_seam_coor = getSeamCoor(vertices, faces, list(set(smpl_segm['rightUpLeg']) & set(smpl_segm['hips'])))
    leftLeg_seam_coor = getSeamCoor(vertices, faces, list(set(smpl_segm['leftLeg']) & set(smpl_segm['leftUpLeg'])))
    rightLeg_seam_coor = getSeamCoor(vertices, faces, list(set(smpl_segm['rightLeg']) & set(smpl_segm['rightUpLeg'])))
    leftFoot_seam_coor = getSeamCoor(vertices, faces, list(set(smpl_segm['leftFoot']) & set(smpl_segm['leftLeg'])))
    rightFoot_seam_coor = getSeamCoor(vertices, faces, list(set(smpl_segm['rightFoot']) & set(smpl_segm['rightLeg'])))

    # Get body part coordinate
    head_coor = getBPartTriCoor(vertices, faces, smpl_segm['head'])
    spineUp_coor = getBPartTriCoor(vertices, faces, smpl_segm['spineUp'])
    spine_coor = getBPartTriCoor(vertices, faces, smpl_segm['spine'])
    hips_coor = getBPartTriCoor(vertices, faces, smpl_segm['hips'])
    leftArm_coor = getBPartTriCoor(vertices, faces, smpl_segm['leftArm'])
    rightArm_coor = getBPartTriCoor(vertices, faces, smpl_segm['rightArm'])
    leftForeArm_coor = getBPartTriCoor(vertices, faces, smpl_segm['leftForeArm'])
    rightForeArm_coor = getBPartTriCoor(vertices, faces, smpl_segm['rightForeArm'])
    leftHand_coor = getBPartTriCoor(vertices, faces, smpl_segm['leftHand'])
    rightHand_coor = getBPartTriCoor(vertices, faces, smpl_segm['rightHand'])
    leftUpLeg_coor = getBPartTriCoor(vertices, faces, smpl_segm['leftUpLeg'])
    rightUpLeg_coor = getBPartTriCoor(vertices, faces, smpl_segm['rightUpLeg'])
    leftLeg_coor = getBPartTriCoor(vertices, faces, smpl_segm['leftLeg'])
    rightLeg_coor = getBPartTriCoor(vertices, faces, smpl_segm['rightLeg'])
    leftFoot_coor = getBPartTriCoor(vertices, faces, smpl_segm['leftFoot'])
    rightFoot_coor = getBPartTriCoor(vertices, faces, smpl_segm['rightFoot'])

    # Calculate volume    
    head_vol = getPartVolume(head_coor) + getPartVolume(neck_seam_coor)
    spineUp_vol = getPartVolume(spineUp_coor) + getPartVolume(neck_seam_coor) \
                    + getPartVolume(leftShoulder_seam_coor) + getPartVolume(rightShoulder_seam_coor) \
                    + getPartVolume(midSpine_seam_coor)
    spine_vol = getPartVolume(spine_coor) + getPartVolume(midSpine_seam_coor) \
                    + getPartVolume(hips_seam_coor)
    hips_vol = getPartVolume(hips_coor) + getPartVolume(hips_seam_coor) \
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

    return {'head': head_vol, 'spineUp': spineUp_vol, 'spine': spine_vol,
            'hips': hips_vol, 'leftUpLeg': leftUpLeg_vol, 'rightUpLeg': rightUpLeg_vol,
            'leftLeg': leftLeg_vol, 'rightLeg': rightLeg_vol,
            'leftFoot': leftFoot_vol, 'rightFoot': rightFoot_vol,
            'leftArm': leftArm_vol, 'rightArm': rightArm_vol,
            'leftForeArm': leftForeArm_vol, 'rightForeArm': rightForeArm_vol,
            'leftHand': leftHand_vol, 'rightHand': rightHand_vol}

def getPartSegm(config):

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
        spine1          267     
        spine2          615     
        head            1194
        neck            156
        hips            487     //

        Paper 16 parts
        -----
        head + neck     Hand + HandIndex1
        Shoulder + Arm  Foot + ToeBase
        spine2 + spine1
    '''

    # Read JSON segmentation file
    segm_path = config.body_part_segm.smpl_path
    with open(segm_path) as json_file:
        smpl_segm = json.load(json_file)

    # Union body parts (from 24 to 16)
    head_segm = list(set.union(set(smpl_segm['head']), set(smpl_segm['neck'])))
    leftHand_segm = list(set.union(set(smpl_segm['leftHand']), set(smpl_segm['leftHandIndex1'])))
    rightHand_segm = list(set.union(set(smpl_segm['rightHand']), set(smpl_segm['rightHandIndex1'])))
    leftArm_segm = list(set.union(set(smpl_segm['leftShoulder']), set(smpl_segm['leftArm'])))
    rightArm_segm = list(set.union(set(smpl_segm['rightShoulder']), set(smpl_segm['rightArm'])))
    leftFoot_segm = list(set.union(set(smpl_segm['leftFoot']), set(smpl_segm['leftToeBase'])))
    rightFoot_segm = list(set.union(set(smpl_segm['rightFoot']), set(smpl_segm['rightToeBase'])))
    spineUp_segm = list(set.union(set(smpl_segm['spine1']), set(smpl_segm['spine2'])))

    return {'head': head_segm, 'spineUp': spineUp_segm, 'spine': smpl_segm['spine'],
            'hips': smpl_segm['hips'], 'leftArm': leftArm_segm, 'rightArm': rightArm_segm,
            'leftForeArm': smpl_segm['leftForeArm'], 'rightForeArm': smpl_segm['rightForeArm'],
            'leftHand': leftHand_segm, 'rightHand': rightHand_segm,
            'leftUpLeg': smpl_segm['leftUpLeg'], 'rightUpLeg': smpl_segm['rightUpLeg'],
            'leftLeg': smpl_segm['leftLeg'], 'rightLeg': smpl_segm['rightLeg'],
            'leftFoot': leftFoot_segm, 'rightFoot': rightFoot_segm}

def getSeamCoor(vertices, faces, vseam):
    # coordinate of centroid
    centroid = np.zeros(3)
    triangles = []

    # Find centroid and neighbors between each vertices
    for vertex in vseam:
        centroid[0] += vertices[vertex][0]
        centroid[1] += vertices[vertex][1]
        centroid[2] += vertices[vertex][2]

        neighbors = set()
        # Find neighbors in all triangle face
        for triangle in faces:
            if vertex in triangle and len(set(vseam) & set(triangle)) == 2:
                neighbors.update(set(vseam) & set(triangle))

        # Create triangle of seam 
        for v in neighbors:
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

    coor = np.array(coor)
    return coor

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
    coors = np.array(coors)
    return coors
        
def getSignedVolume(pi, pj, pk):
    vkji = pk[0]*pj[1]*pi[2]
    vjki = pj[0]*pk[1]*pi[2]
    vkij = pk[0]*pi[1]*pj[2]
    vikj = pi[0]*pk[1]*pj[2]
    vjik = pj[0]*pi[1]*pk[2]
    vijk = pi[0]*pj[1]*pk[2]
    return (1.0/6.0)*(-vkji+vjki+vkij-vikj-vjik+vijk)

def getPartVolume(triangle_coor):
    vol = 0
    for tri in triangle_coor:
        vol += getSignedVolume(tri[0], tri[1], tri[2])
    return vol

def getVolumeDirection(source_vol, target_vol, offsetSource, smpl_segm):
    vDirection = np.zeros((len(offsetSource), 3))
    for part in smpl_segm:
        for id in smpl_segm[part]:
            vDirection[id] = (target_vol[part] - source_vol[part])*offsetSource[id][0]
    return vDirection