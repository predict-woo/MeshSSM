# # Copyright (c) 2023, Tri Dao, Albert Gu.

import torch
import igl
from pytorch3d.structures import Meshes
from einops import rearrange
from pytorch3d.io import load_obj, save_obj

device = torch.device("cuda:6")


def igl_compute_adjacencies(F):
    F = F.cpu().numpy()
    TT, TTi = igl.triangle_triangle_adjacency(F)
    TT = torch.tensor(TT, dtype=torch.int32, device=device)
    ind = torch.arange(TT.shape[0], device=device).unsqueeze(-1).repeat(1, TT.shape[1])
    TT_combined = torch.stack((ind, TT), dim=-1)
    edges = TT_combined.reshape(-1, 2)
    edges = edges[edges[:, 1] != -1]
    edges = torch.sort(edges, dim=1)[0]
    unique_edges, inverse_indices, counts = torch.unique(
        edges, return_inverse=True, return_counts=True, dim=0
    )

    return unique_edges


def igl_conv_torch_compute_adjacencies(F):
    n = F.max().item() + 1
    VF = [[] for _ in range(n)]
    NI = torch.zeros(n + 1, dtype=torch.int32, device=device)
    TT = torch.full((F.size(0), 3), -1, dtype=torch.int32, device=device)

    # Create vertex to triangle adjacency
    for f in range(F.size(0)):
        for k in range(3):
            VF[F[f, k].item()].append(f)
    NI[1:] = torch.cumsum(
        torch.tensor([len(v) for v in VF], dtype=torch.int32, device=device), dim=0
    )

    # Flatten the VF list
    VF_flat = torch.zeros(NI[-1].item(), dtype=torch.int32, device=device)
    idx = 0
    for v in VF:
        VF_flat[idx : idx + len(v)] = torch.tensor(v, dtype=torch.int32, device=device)
        idx += len(v)

    # Compute TT
    for f in range(F.size(0)):
        for k in range(3):
            vi = F[f, k].item()
            vin = F[f, (k + 1) % 3].item()
            for j in range(NI[vi], NI[vi + 1]):
                fn = VF_flat[j].item()
                if fn != f and (F[fn, 0] == vin or F[fn, 1] == vin or F[fn, 2] == vin):
                    TT[f, k] = fn
                    break

    ind = torch.arange(TT.shape[0], device=device).unsqueeze(-1).repeat(1, TT.shape[1])
    TT_combined = torch.stack((ind, TT), dim=-1)
    edges = TT_combined.reshape(-1, 2)
    edges = edges[edges[:, 1] != -1]
    edges = torch.sort(edges, dim=1)[0]
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
    unique_edges, inverse_indices, counts = torch.unique(
        edges, return_inverse=True, return_counts=True, dim=0
    )
    # Filter to get the edges that are shared by exactly two faces
    shared_edge_indices = (counts == 2).nonzero(as_tuple=True)[0]
    # rearrange unique edges to be (b, n, v)
    inverse_indices = rearrange(inverse_indices, "(b n) -> b n", n=3)
    # check if the edge is shared by two faces
    mask = inverse_indices.unsqueeze(-1) == shared_edge_indices.unsqueeze(0)
    # get indices of faces that share the edge
    indices = mask.nonzero(as_tuple=False)
    # sort the indices based on the second index
    sorted_tensor, sorted_indices = torch.sort(indices[:, 2])
    sorted_tensor = indices[sorted_indices]
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


verts, faces, aux = load_obj("tea.txt")
meshes = Meshes(verts=[verts] * 100, faces=[faces.verts_idx] * 100)
meshes.to(device)

import time

# start = time.time()
# TT_2 = compute_adjacencies(meshes.faces_packed())
# TT_2 = lexical_sort(TT_2)
# print(time.time() - start)
# print(TT_2, TT_2.shape)


start = time.time()
TT = igl_compute_adjacencies(meshes.faces_packed())
TT = lexical_sort(TT)
print(time.time() - start)
print(TT, TT.shape)
