from omegaconf import OmegaConf
from dataclasses import dataclass
from typing import Tuple
from dataclasses import dataclass

@dataclass
class Variable:
    create: bool = True
    requires_grad: bool = True


@dataclass
class Pose(Variable):
    type: str = 'aa'

@dataclass
class PCA:
    num_comps: int = 12
    flat_hand_mean: bool = False


@dataclass
class PoseWithPCA(Pose):
    pca: PCA = PCA()


@dataclass
class Shape(Variable):
    num: int = 10


@dataclass
class Expression(Variable):
    num: int = 10


@dataclass
class SMPL:
    betas: Shape = Shape()
    global_rot: Pose = Pose()
    body_pose: Pose = Pose()
    translation: Variable = Variable()


@dataclass
class SMPLH(SMPL):
    left_hand_pose: PoseWithPCA = PoseWithPCA()
    right_hand_pose: PoseWithPCA = PoseWithPCA()


@dataclass
class SMPLX(SMPLH):
    expression: Expression = Expression()
    jaw_pose: Pose = Pose()
    leye_pose: Pose = Pose()
    reye_pose: Pose = Pose()


@dataclass
class MANO:
    betas: Shape = Shape()
    wrist_pose: Pose = Pose()
    hand_pose: PoseWithPCA = PoseWithPCA()
    translation: Variable = Variable()


@dataclass
class FLAME:
    betas: Shape = Shape()
    expression: Expression = Expression()
    global_rot: Pose = Pose()
    neck_pose: Pose = Pose()
    jaw_pose: Pose = Pose()
    leye_pose: Pose = Pose()
    reye_pose: Pose = Pose()


@dataclass
class BodyModelConfig:
    model_type: str = 'smplx'
    use_compressed: bool = True
    folder: str = 'models'
    gender: str = 'neutral'
    extra_joint_path: str = ''
    ext: str = 'npz'

    num_expression_coeffs: int = 10

    use_face_contour: bool = True
    joint_regressor_path: str = ''

    smpl: SMPL = SMPL()
    star: SMPL = SMPL()
    smplh: SMPLH = SMPLH()
    smplx: SMPLX = SMPLX()
    mano: MANO = MANO()
    flame: FLAME = FLAME()


conf = OmegaConf.structured(BodyModelConfig)