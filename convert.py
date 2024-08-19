import os
import torch
import numpy as np


# torch device cpu


def convert_pt_to_npy(input_dir, output_dir):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop over all files in the input directory
    for filename in os.listdir(input_dir):
        print(filename)
        if filename.endswith(".pt"):
            # Construct full file path
            pt_file_path = os.path.join(input_dir, filename)
            print(pt_file_path)

            # Load the tensor from the .pt file
            tensor = torch.load(pt_file_path)
            print(tensor.shape)

            # Convert the tensor to a NumPy array
            np_array = tensor.cpu().detach().numpy()

            # Create the output file path with .npy extension
            npy_filename = os.path.splitext(filename)[0] + ".npy"
            npy_file_path = os.path.join(output_dir, npy_filename)

            # Save the NumPy array as a .npy file
            np.save(npy_file_path, np_array)

            print(f"Converted {filename} to {npy_filename}")


# Example usage
input_directory = "chair_encoded_dataset"  # Replace with your input directory path
output_directory = "chair_encoded_dataset_np"

convert_pt_to_npy(input_directory, output_directory)
