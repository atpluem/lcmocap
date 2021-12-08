from networkx.classes.function import neighbors
from numpy.core.numeric import indices
import torch
import numpy as np
import trimesh
import networkx as nx

from mesh_viewer import MeshViewer
from scipy import sparse

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

    def run_fitting(self):
        
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

    ##############################################################
    ##                  Laplacian Mesh Editing                  ##
    ##############################################################
    """
        Parameters
        ----------
        vertices : ndarray (N, 3) 
            Array of vertex positions
        faces : ndarray (M, 3)
            Array of triangle indices

        Returns
        -------
        L : scipy.sparse (NVertices, NVertices)
            A sparse Laplacian matrix with cotangent weights
    """

    def get_LaplacianMatrixUmbrella(self, vertices, faces):

        N = vertices.shape[0]
        M = faces.shape[0]

        I = np.zeros(M*6)
        J = np.zeros(M*6)
        V = np.ones(M*6)

        # Set up umbrella entries
        for shift in range(3):
            [i, j, k] = [shift, (shift+1)%3, (shift+2)%3]
            I[shift*M*2:shift*M*2+M] = faces[:, i]
            J[shift*M*2:shift*M*2+M] = faces[:, j]
            I[shift*M*2+M:shift*M*2+2*M] = faces[:, j]
            J[shift*M*2+M:shift*M*2+2*M] = faces[:, i]

        # Create laplacian matrix
        L = sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()
        L[L > 0] = 1
        # Create the diagonal by summing the rows and subtracting off the nondiagonal entries
        L = sparse.dia_matrix((L.sum(1).flatten(), 0), L.shape) - L

        print(L)    
        return L


    def get_LaplacianMatrixCotangent(self, vertices, faces):
        
        N = vertices.shape[0]
        M = faces.shape[0]

        I = np.zeros(M*6)
        J = np.zeros(M*6)
        V = np.zeros(M*6)

        IA = np.zeros(M*3)
        VA = np.zeros(M*3) 
        VC = 1.0*np.ones(M*3)

        # Compute cotangent weights
        for shift in range(3):
            # Compute different cotangent weights
            [i, j, k] = [shift, (shift+1)%3, (shift+2)%3]
            dV1 = vertices[faces[:, i], :] - vertices[faces[:, k], :]
            dV2 = vertices[faces[:, j], :] - vertices[faces[:, k], :]
            Normal = np.cross(dV1, dV2)

            # Cotangent (dot product / magnitude cross product)
            NMag = np.sqrt(np.sum(Normal**2, 1))
            cotAlpha = np.sum(dV1*dV2, 1)/NMag
            I[shift*M*2:shift*M*2+M] = faces[:, i]
            J[shift*M*2:shift*M*2+M] = faces[:, j]
            V[shift*M*2:shift*M*2+M] = cotAlpha
            I[shift*M*2+M:shift*M*2+2*M] = faces[:, j]
            J[shift*M*2+M:shift*M*2+2*M] = faces[:, i]
            V[shift*M*2+M:shift*M*2+2*M] = cotAlpha
            
            if shift == 0:
                # Compute contribution of this triangle to each of the vertices
                for k in range(3):
                    IA[k*M:(k+1)*M] = faces[:, k]
                    VA[k*M:(k+1)*M] = 0.5*NMag

        # Create laplacian matrix
        L = sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()
        # Create the diagonal by summing the rows and subtracting off the nondiagonal entries
        L = sparse.dia_matrix((L.sum(1).flatten(), 0), L.shape) - L
        
        return L
