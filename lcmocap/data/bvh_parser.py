import torch
import numpy as np

from utils import BVH_mod as BVH
from utils.Quaternions import Quaternions
from models.Kinematics import ForwardKinematics
from models.skeleton import build_edge_topology

"""
1.
Specify the joints that you want to use in training and test. Other joints will be discarded.
Please start with root joint, then left leg chain, right leg chain, head chain, left shoulder chain and right shoulder chain.
See the examples below.
"""
corps_name_1 = ['Pelvis', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand']
corps_name_smpl = ['Hips', 
                   'LeftUpLeg', 'LeftLeg', 'LeftFoot',  
                   'RightUpLeg', 'RightLeg', 'RightFoot', 
                   'Spine', 'SpinUp', 
                   'Head', 
                   'LeftArm', 'LeftForeArm', 'LeftHand', 
                   'RightArm', 'RightForeArm', 'RightHand']


"""
2.
Specify five end effectors' name.
Please follow the same order as in 1.
"""
ee_name_1 = ['LeftToeBase', 'RightToeBase', 'Head', 'LeftHand', 'RightHand']
ee_name_smpl = ['LeftFoot, RightFoot', 'Head', 'LeftHand', 'RightHand']

corps_names = [corps_name_1, corps_name_smpl]
ee_names = [ee_name_1, ee_name_smpl]

"""
3.
Add previously added corps_name and ee_name at the end of the two above lists.
"""
class BVH_file:
    def __init__(self, file_path=None, args=None, dataset=None, new_root=None):
        self.anim, self._names, self.frametime = BVH.load(file_path)
        if new_root is not None:
            self.set_new_root(new_root)
        self.skeleton_type = -1
        self.edges = []
        self.edge_mat = []
        self.edge_num = 0
        self._topology = None
        self.ee_length = []

        for i, name in enumerate(self._names):
            if ':' in name:
                name = name[name.find(':') + 1]
                self._names[i] = name

        full_fill = [1] * len(corps_names)
        for i, ref_names in enumerate(corps_names):
            for ref_name in ref_names:
                if ref_name not in self._names:
                    full_fill[i] = 0
                    break

        for i, _ in enumerate(full_fill):
            if full_fill[i]:
                self.skeleton_type = i
                break 

        """
        4. 
        Here, you need to assign self.skeleton_type the corresponding index of your own dataset in corps_names or ee_names list.
        You can use self._names, which contains the joints name in original bvh file, to write your own if statement.
        """

        if self.skeleton_type == -1:
            print(self._names)
            raise Exception('Unknow skeleton')
        
        if self.skeleton_type == 0:
            self.set_new_root(1)

        # Virtual JOINT
        self.details = [i for i, name in enumerate(self._names) \
                        if name not in corps_names[self.skeleton_type]]
        self.joint_num = self.anim.shape[1]
        self.corps = []
        self.simplified_name = []
        self.simplify_map = {}
        self.inverse_simplify_map = {}

        # Assign corp ID
        for name in corps_names[self.skeleton_type]:
            for j in range(self.anim.shape[1]):
                if name == self._names[j]:
                    self.corps.append(j)
                    break
         
        if len(self.corps) != len(corps_names[self.skeleton_type]):
            for i in self.corps:
                print(self._names[i], end=' ')
            print(self.corps, self.skeleton_type, len(self.corps), sep='\n')
            raise Exception('Problem in file', file_path)

        # Assign effectors ID
        self.ee_id = []
        for i in ee_names[self.skeleton_type]:
            self.ee_id.append(corps_names[self.skeleton_type].index(i))

        # Addign simplify_map, inverse_simplify_map dict
        self.joint_num_simplify = len(self.corps)
        for i, j in enumerate(self.corps):
            self.simplify_map[j] = i
            self.inverse_simplify_map[i] = j
            self.simplified_name.append(self._names[j])
        self.inverse_simplify_map[0] = -1
        for i in range(self.anim.shape[1]):
            if i in self.details:
                self.simplify_map[i] = -1
        
        self.edges = build_edge_topology(self.topology, self.offset)

    def scale(self, alpha):
        self.anim.offsets *= alpha
        global_position = self.anim.positions[:, 0, :]
        global_position[1:, :] *= alpha
        global_position[1:, :] += (1 - alpha) * global_position[0, :]

    def rotate(self, theta, axis):
        q = Quaternions(np.hstack((np.cos(theta/2), np.sin(theta/2) * axis)))
        position = self.anim.positions[:, 0, :].copy()
        rotation = self.anim.rotations[:, 0, :]
        position[1:, ...] -= position[0:-1, ...]
        q_position = Quaternions(np.hstack((np.zeros((position.shape[0], 1)), position)))
        q_rotation = Quaternions.from_euler(np.radians(rotation))
        q_rotation = q * q_rotation
        q_position = q * q_position * (-q)
        self.anim.rotations[:, 0, :] = np.degrees(q_rotation.euler())
        position = q_position.imaginaries
        for i in range(1, position.shape[0]):
            position[i] += position[i-1]
        self.anim.positions[:, 0, :] = position

    @property
    def topology(self):
        if self._topology is None:
            self._topology = self.anim.parents[self.corps].copy()
            for i in range(self._topology.shape[0]):
                if i >= 1: self._topology[i] = self.simplify_map[self._topology[i]]
            self._topology = tuple(self._topology)
        return self._topology

    def get_ee_id(self):
        return self.ee_id

    def to_numpy(self, quater=False, edge=True):
        rotations = self.anim.rotations[:, self.corps, :]
        if quater:
            rotations = Quaternions.from_euler(np.radians(rotations)).qs
            positions = self.anim.positions[:, 0, :]
        else:
            positions = self.anim.positions[:, 0, :]
        if edge:
            index = []
            for e in self.edges:
                index.append(e[0])
            rotations = rotations[:, index, :]

        rotations = rotations.reshape(rotations.shape[0], -1)

        return np.concatenate((rotations, positions), axis=1)
    
    def to_tensor(self, quater=False, edge=True):
        res = self.to_numpy(quater, edge)
        res = torch.tensor(res, dtype=torch.float)
        res = res.permute(1, 0)
        res = res.reshape((-1, res.shape[-1]))
        return res

    def get_position(self):
        positions = self.anim.positions
        positions = positions[:, self.corps, :]
        return positions

    @property
    def offset(self):
        return self.anim.offsets[self.corps]

    @property
    def names(self):
        return self.simplified_name

    def get_height(self):
        offset = self.offset
        topo = self.topology

        res = 0
        p = self.ee_id[0]
        while p != 0:
            res += np.dot(offset[p], offset[p]) ** 0.5
            p = topo[p]

        p = self.ee_id[2]
        while p != 0:
            res += np.dot(offset[p], offset[p]) ** 0.5
            p = topo[p]

        return res

    def write(self, file_path):
        motion = self.to_numpy(quater=False, edge=False)
        rotations = motion[..., :-3].reshape(motion.shape[0], -1, 3)
        positions = motion[..., -3:]
        write_bvh(self.topology, self.offset, rotations, positions, self.names, 1.0/30, 'xyz', file_path)

    def get_ee_length(self):
        if len(self.ee_length): return self.ee_length
        degree = [0] * len(self.topology)
        for i in self.topology:
            if i < 0: continue
            degree[i] += 1

        for j in self.ee_id:
            length = 0
            while degree[j] <= 1:
                t = self.offset[j]
                length += np.dot(t, t) ** 0.5
                j = self.topology[j]

            self.ee_length.append(length)

        height = self.get_height()
        ee_group = [[0, 1], [2], [3, 4]]
        for group in ee_group:
            maxv = 0
            for j in group:
                maxv = max(maxv, self.ee_length[j])
            for j in group:
                self.ee_length[j] *= height / maxv

        return self.ee_length

    def set_new_root(self, new_root):
        euler = torch.tensor(self.anim.rotations[:, 0, :], dtype=torch.float)
        transform = ForwardKinematics.transform_from_euler(euler, 'xyz')
        offset = torch.tensor(self.anim.offsets[new_root], dtype=torch.float)
        new_pos = torch.matmul(transform, offset)
        new_pos = new_pos.numpy() + self.anim.positions[:, 0, :]
        self.anim.offsets[0] = -self.anim.offsets[new_root]
        self.anim.offsets[new_root] = np.zeros((3, ))
        self.anim.positions[:, new_root, :] = new_pos
        rot0 = Quaternions.from_euler(np.radians(self.anim.rotations[:, 0, :]), order='xyz')
        rot1 = Quaternions.from_euler(np.radians(self.anim.rotations[:, new_root, :]), order='xyz')
        new_rot1 = rot0 * rot1
        new_rot0 = (-rot1)
        new_rot0 = np.degrees(new_rot0.euler())
        new_rot1 = np.degrees(new_rot1.euler())
        self.anim.rotations[:, 0, :] = new_rot0
        self.anim.rotations[:, new_root, :] = new_rot1

        new_seq = []
        vis = [0] * self.anim.rotations.shape[1]
        new_idx = [-1] * len(vis)
        new_parent = [0] * len(vis)

        def relabel(x):
            nonlocal new_seq, vis, new_idx, new_parent
            new_idx[x] = len(new_seq)
            new_seq.append(x)
            vis[x] = 1
            for y in range(len(vis)):
                if not vis[y] and (self.anim.parents[x] == y or self.anim.parents[y] == x):
                    relabel(y)
                    new_parent[new_idx[y]] = new_idx[x]

        relabel(new_root)
        self.anim.rotations = self.anim.rotations[:, new_seq, :]
        self.anim.offsets = self.anim.offsets[new_seq]
        names = self._names.copy()
        for i, j in enumerate(new_seq):
            self._names[i] = names[j]
        self.anim.parents = np.array(new_parent, dtype=np.int)