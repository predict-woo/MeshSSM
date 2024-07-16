import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data

class GraphConvEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int,
        normalize: bool = True,
        project: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Create a list of SAGEConv layers
        self.conv_layers = nn.ModuleList()
        
        # Input layer
        self.conv_layers.append(SAGEConv(input_dim, hidden_dims[0], normalize=normalize, project=project))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.conv_layers.append(SAGEConv(hidden_dims[i], hidden_dims[i+1], normalize=normalize, project=project))
            # BatchNorm
        
        # Output layer
        self.conv_layers.append(SAGEConv(hidden_dims[-1], output_dim, normalize=normalize, project=project))
        
        # Activation function
        self.activation = nn.SiLU()
        
        # Layer normalization
        self.norm = nn.LayerNorm(output_dim)
        

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass of the GraphConvEncoder.
        
        Args:
            data (Data): A PyTorch Geometric Data object containing:
                - x (torch.Tensor): Node feature matrix
                - edge_index (torch.Tensor): Graph connectivity in COO format
        
        Returns:
            torch.Tensor: Encoded representation of the input graph
        """
        x, edge_index = data.x, data.edge_index
        
        # Apply graph convolutions
        for conv in self.conv_layers[:-1]:
            x = self.activation(conv(x, edge_index))
        
        # Final convolution
        x = self.conv_layers[-1](x, edge_index)
        
        # Apply normalization
        x = self.norm(x)
        
        return x

# Example usage
if __name__ == "__main__":
    # Define model parameters
    input_dim = 196  # Assuming 3D coordinates as input features
    hidden_dims = [64, 128, 256]
    output_dim = 512
    
    # Create the model
    model = GraphConvEncoder(input_dim, hidden_dims, output_dim)
    
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