# # Copyright (c) 2023, Tri Dao, Albert Gu.

import torch
import numpy as np
import igl
from pytorch3d.structures import Meshes
from einops import rearrange
from pytorch3d.io import load_obj

# Example mesh provided by the user
verts = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float)
faces = torch.tensor(
    [
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 1],
    ],
    dtype=torch.int64,
)
meshes = Meshes(verts=[verts], faces=[faces])


# Function to compute the triangle-triangle adjacency matrix
def compute_triangle_adjacency(meshes: Meshes):
    # Extract vertices and faces from the PyTorch3D mesh
    verts = meshes.verts_list()[0]  # Assuming a single mesh in the list
    faces = meshes.faces_list()[0]

    # Convert to numpy arrays
    verts_np = verts.numpy()
    faces_np = faces.numpy()

    # Compute the adjacency matrix using libigl
    TT, TTi = igl.triangle_triangle_adjacency(faces_np)

    # TT to tensor
    TT = torch.tensor(TT, dtype=torch.int64)

    # create index matrix of same size
    ind = torch.arange(TT.shape[0]).unsqueeze(-1).repeat(1, TT.shape[1])

    # append ind to each element in TT
    TT_combined = torch.stack((ind, TT), dim=-1)

    # flatten and reshape
    edges = TT_combined.reshape(-1, 2)
    edges = edges[edges[:, 1] != -1]

    # sort edges
    edges = torch.sort(edges, dim=1)[0]

    # unique edges
    unique_edges, inverse_indices, counts = torch.unique(
        edges, return_inverse=True, return_counts=True, dim=0
    )

    return unique_edges


def compute_adjacencies(faces):
    # rotate vertices in faces
    rotated_faces = torch.roll(faces, 1, dims=1)
    # concat to create edges
    concat_faces = torch.cat([faces.unsqueeze(-1), rotated_faces.unsqueeze(-1)], dim=-1)
    # flatten to get a list of edges
    edges = rearrange(concat_faces, "b n v -> (b n) v")
    # Sort edges so that the smaller index comes first
    edges = torch.sort(edges, dim=1)[0]
    # get unique edges
    print(edges)
    unique_edges, inverse_indices, counts = torch.unique(
        edges, return_inverse=True, return_counts=True, dim=0
    )
    print(unique_edges)
    print(inverse_indices)
    print(counts)
    # Filter to get the edges that are shared by exactly two faces
    shared_edge_indices = (counts == 2).nonzero(as_tuple=True)[0]
    print(shared_edge_indices)
    # check if the edge is shared by two faces
    mask = inverse_indices.unsqueeze(1) == shared_edge_indices.unsqueeze(0)
    print(mask)
    # get indices of faces that share the edge
    indices = mask.nonzero(as_tuple=False)
    print(indices)
    # sort the indices based on the second index
    sorted_tensor, sorted_indices = torch.sort(indices[:, 1])
    sorted_tensor = indices[sorted_indices]
    # get the first index
    sorted_tensor = sorted_tensor[:, 0]
    # reshape to get a list of edges
    adjacencies = sorted_tensor.view(-1, 2)

    return adjacencies


def compute_adjacencies(faces):
    # rotate vertices in faces
    rotated_faces = torch.roll(faces, 1, dims=1)
    # concat to create edges
    concat_faces = torch.cat([faces.unsqueeze(-1), rotated_faces.unsqueeze(-1)], dim=-1)
    # flatten to get a list of edges
    edges = rearrange(concat_faces, "b n v -> (b n) v")
    # Sort edges so that the smaller index comes first
    edges = torch.sort(edges, dim=1)[0]
    # get unique edges
    print(edges)
    unique_edges, inverse_indices, counts = torch.unique(
        edges, return_inverse=True, return_counts=True, dim=0
    )
    print(unique_edges)
    print(inverse_indices)
    print(counts)
    # Filter to get the edges that are shared by exactly two faces
    shared_edge_indices = (counts == 2).nonzero(as_tuple=True)[0]
    print(shared_edge_indices)
    # rearrange unique edges to be (b, n, v)
    inverse_indices = rearrange(inverse_indices, "(b n) -> b n", n=3)
    print("inverse_indices", inverse_indices)
    # check if the edge is shared by two faces
    mask = inverse_indices.unsqueeze(-1) == shared_edge_indices.unsqueeze(0)
    print(mask, mask.shape)
    # get indices of faces that share the edge
    indices = mask.nonzero(as_tuple=False)
    print(indices)
    # sort the indices based on the second index
    sorted_tensor, sorted_indices = torch.sort(indices[:, 2])
    sorted_tensor = indices[sorted_indices]
    print(sorted_tensor)
    # get the first index
    sorted_tensor = sorted_tensor[:, 0]
    # reshape to get a list of edges
    adjacencies = sorted_tensor.view(-1, 2)

    adjacencies = adjacencies.sort(dim=1)[0]

    return adjacencies


def lexical_sort(tensor):
    # sort by 1th index in 1th dimension
    _, first_ind = torch.sort(tensor[:, 1], stable=True)
    tensor = tensor[first_ind]

    # # sort by 0th index in 0th dimension
    _, zero_ind = torch.sort(tensor[:, 0], stable=True)
    tensor = tensor[zero_ind]

    return tensor


verts, faces, aux = load_obj("Horse.obj")
meshes = Meshes(verts=[verts], faces=[faces.verts_idx])


TT_2 = compute_adjacencies(faces.verts_idx)
TT_2 = lexical_sort(TT_2)
print("Triangle-Triangle Adjacency Matrix:\n", TT_2)

# print entire tensor to file
with open("TT_2.txt", "w") as f:
    for i in range(TT_2.shape[0]):
        f.write(str(TT_2[i][0].item()) + " " + str(TT_2[i][1].item()) + "\n")


# # Compute the adjacency matrix
TT = compute_triangle_adjacency(meshes)
TT = lexical_sort(TT)
print("Triangle-Triangle Adjacency Matrix:\n", TT)

with open("TT.txt", "w") as f:
    for i in range(TT.shape[0]):
        f.write(str(TT[i][0].item()) + " " + str(TT[i][1].item()) + "\n")

# # print 1th dimension elements in TT that are not in TT_2
# for i in range(TT.shape[0]):
#     if TT[i, 0] != TT_2[i, 0] or TT[i, 1] != TT_2[i, 1]:
#         print(TT[i])
#         print(TT_2[i])
#         exit()


# TT = compute_adjacencies(faces)
# print("Triangle-Triangle Adjacency Matrix:\n", TT)

# TT = compute_triangle_adjacency(meshes)
# print("Triangle-Triangle Adjacency Matrix:\n", TT)
