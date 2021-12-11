import trimesh
import numpy as np
import networkx as nx

def getShapeOffset(vertices, faces):
    neighbors = getNeighbors(vertices, faces)
    offset = getLaplacianOffset(vertices, neighbors)
    return offset
    
def getNeighbors(vertices, faces):
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    graph = nx.from_edgelist(mesh.edges_unique)
    neighbors = [list(graph[i].keys()) for i in range(len(vertices))]
    return np.array(neighbors)

def getLaplacianOffset(vertices, neighbors):
    offset = np.zeros((len(neighbors), 3))
    for id in range(len(neighbors)):
        indices = len(neighbors[id])
        for indice in range(indices):
            offset[id] = offset[id] \
                + (vertices[neighbors[id][indice]]/indices - vertices[id])
    return offset

def getOptimalPosition(vertices, neighbors, offsetSource, offsetTarget):
    newVertices = np.zeros((len(offsetSource), 3))
    for id in range(len(offsetSource)):
        indices = len(neighbors[id])
        for indice in range(indices):
            newVertices[id] = newVertices[id] + \
                (vertices[neighbors[id][indice]]/indices) \
                - (offsetTarget[id][0]*offsetSource[id][0] \
                + offsetTarget[id][1]*offsetSource[id][1] \
                + offsetTarget[id][2]*offsetSource[id][2])
    return newVertices