import trimesh
import numpy as np
import networkx as nx

from scipy import sparse
    
def getNeighbors(vertices, faces):
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    graph = nx.from_edgelist(mesh.edges_unique)
    neighbors = [list(graph[i].keys()) for i in range(len(vertices))]
    return np.array(neighbors)

def getLaplacianOffset(vertices, neighbors):
    offset = np.zeros((len(vertices), 3))
    for id in range(len(vertices)):
        indices = len(neighbors[id])
        for indice in range(indices):
            offset[id] += vertices[neighbors[id][indice]]/indices
        offset[id] -= vertices[id]
    return offset

def getShapeDirection(vertices, neighbors, offsetSource, offsetTarget):
    sDirection = np.zeros((len(vertices), 3))
    for id in range(len(vertices)):
        indices = len(neighbors[id])
        for indice in range(indices):
            sDirection[id] += (vertices[neighbors[id][indice]]/indices)
        sDirection[id] -= (offsetTarget[id][0]*offsetSource[id][0] \
                        + offsetTarget[id][1]*offsetSource[id][1] \
                        + offsetTarget[id][2]*offsetSource[id][2])
    return sDirection

def get_LaplacianMatrixUmbrella(vertices, faces):
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
    return L

def get_LaplacianMatrixCotangent(vertices, faces):
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