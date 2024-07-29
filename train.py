import torch
import torch.nn as nn
from models.mesh_encoder import MeshEncoder
from utils.mesh import FaceFeatureExtractor
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
