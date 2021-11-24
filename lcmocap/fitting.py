from numpy.lib.function_base import append
import torch
import numpy as np

from mesh_viewer import MeshViewer

@torch.no_grad()
def guess_init():
    pass

class FittingMonitor(object):
    def __init__(self, summary_steps=1, visualize=False,
                 maxiters=100, ftol=2e-09, gtol=1e-05,
                 body_color=(1.0, 1.0, 0.9, 1.0),
                 model_type='smpl'):
        super(FittingMonitor, self).__init__()

        self.maxiters = maxiters
        self.ftol = ftol
        self.gtol = gtol

        self.visualize = visualize
        self.summary_steps = summary_steps
        self.body_color = body_color
        self.model_type = model_type

        ''' Parameters weighting (g: gamma)
            Energy term: gshape, gvol, gc
            Contact term: gr, ga
            Ground contact: grg, gag
            Offset weight: epsilon
        '''
        self.gshape = self.gvol = self.gc = 1
        self.gr = self.ga = 1
        self.grg = self.gag = 0.1
        self.eps = 0.3

    def __enter__(self):
        self.steps = 0
        if self.visualize:
            self.mv = MeshViewer(body_color=self.body_color)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if self.visualize:
            self.mv.close_viewer()

    def set_color(self, vertex_color):
        batch_size = self.colors.shape[0]
        self.colors = np.tile(np.array(vertex_color).reshape(1, 3), [batch_size, 1])

    def run_fitting(self, ):
        
        ''' Helper function for running an optimization process
            Parameters
            ----------
                optimizer: torch.optim.Optimizer
                    The PyTorch optimizer object
                closure: function
                    The function used to calculate the gradients
                params: list
                    List containing the parameters that will be optimized
                body_model: nn.Module
                    The body model PyTorch module
                use_vposer: bool
                    Flag on whether to use VPoser (default=True).
                pose_embedding: torch.tensor, BxN
                    The tensor that contains the latent pose variable.
                vposer: nn.Module
                    The VPoser module
            Returns
            -------
                loss: float
                The final loss value
        '''


            