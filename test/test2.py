import torch
import kaolin.ops.mesh as mesh_ops

from mesh_ssm.utils.augment import augment_mesh

# Check CUDA availability
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

# Check CUDA version
if cuda_available:
    cuda_version = torch.version.cuda
    print(f"CUDA version: {cuda_version}")
else:
    print("CUDA is not available")

# Check PyTorch version
torch_version = torch.__version__
print(f"PyTorch version: {torch_version}")

# Check cuDNN version if available
if cuda_available:
    cudnn_version = torch.backends.cudnn.version()
    print(f"cuDNN version: {cudnn_version}")
