import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, BatchNorm
from torch_geometric.data import Data
from mesh_ssm.utils.mesh import FaceFeatureExtractor


class MeshEncoder(nn.Module):
    def __init__(
        self,
        normalize: bool = False,
        project: bool = False,
    ):
        super().__init__()

        self.sage_conv_1 = SAGEConv(
            in_channels=196, out_channels=64, normalize=normalize, project=project
        )
        self.sage_conv_2 = SAGEConv(
            in_channels=64, out_channels=128, normalize=normalize, project=project
        )
        self.sage_conv_3 = SAGEConv(
            in_channels=128, out_channels=256, normalize=normalize, project=project
        )
        self.sage_conv_4 = SAGEConv(
            in_channels=256, out_channels=256, normalize=normalize, project=project
        )
        self.sage_conv_5 = SAGEConv(
            in_channels=256, out_channels=576, normalize=normalize, project=project
        )

        self.batch_norm_1 = nn.BatchNorm1d(64)
        self.batch_norm_2 = nn.BatchNorm1d(128)
        self.batch_norm_3 = nn.BatchNorm1d(256)
        self.batch_norm_4 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()

        self.feature_extractor = FaceFeatureExtractor()

    def forward(self, x, edge_index) -> torch.Tensor:
        x = self.sage_conv_1(x, edge_index)
        x = self.batch_norm_1(x)
        x = self.relu(x)

        x = self.sage_conv_2(x, edge_index)
        x = self.batch_norm_2(x)
        x = self.relu(x)

        x = self.sage_conv_3(x, edge_index)
        x = self.batch_norm_3(x)
        x = self.relu(x)

        x = self.sage_conv_4(x, edge_index)
        x = self.batch_norm_4(x)
        x = self.relu(x)

        x = self.sage_conv_5(x, edge_index)

        return x


# Example usage
if __name__ == "__main__":
    # Define model parameters
    input_dim = 196  # Assuming 3D coordinates as input features
    hidden_dims = [64, 128, 256]
    output_dim = 512

    # Create the model
    model = MeshEncoder()

    # Create sample input data
    num_nodes = 1000
    num_edges = 3000
    x = torch.randn(num_nodes, input_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))

    # Create a PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index)

    # Forward pass
    output = model(data)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output}")
