import torch


def avg_fv_feat(faces, face_features, num_vertices=None):
    r"""Given features assigned for every vertex of every face, computes per-vertex features by
    averaging values across all faces incident each vertex.

    Args:
       faces (torch.LongTensor): vertex indices of faces of a fixed-topology mesh batch with
            shape :math:`(\text{num_faces}, \text{face_size})`.
       face_features (torch.FloatTensor): any features assigned for every vertex of every face, of shape
            :math:`(\text{batch_size}, \text{num_faces}, \text{face_size}, N)`.
       num_vertices (int, optional): number of vertices V (set to max index in faces, if not set)

    Return:
        (torch.FloatTensor): of shape (B, V, 3)
    """
    if num_vertices is None:
        num_vertices = int(faces.max()) + 1

    B = face_features.shape[0]
    V = num_vertices
    F = faces.shape[0]
    FSz = faces.shape[1]
    Nfeat = face_features.shape[-1]
    vertex_features = torch.zeros(
        (B, V, Nfeat), dtype=face_features.dtype, device=face_features.device
    )
    counts = torch.zeros((B, V), dtype=face_features.dtype, device=face_features.device)

    faces = faces.unsqueeze(0).repeat(B, 1, 1)
    fake_counts = torch.ones(
        (B, F), dtype=face_features.dtype, device=face_features.device
    )
    #              B x F          B x F x 3
    # self[index[i][j][k]][j][k] += src[i][j][k]  # if dim == 0
    # self[i][index[i][j][k]][k] += src[i][j][k]  # if dim == 1
    for i in range(FSz):
        vertex_features.scatter_add_(
            1, faces[..., i : i + 1].repeat(1, 1, Nfeat), face_features[..., i, :]
        )
        counts.scatter_add_(1, faces[..., i], fake_counts)

    counts = counts.clip(min=1).unsqueeze(-1)
    vertex_normals = vertex_features / counts
    return vertex_normals
