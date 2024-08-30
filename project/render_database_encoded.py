import os
from PIL import Image
import igl
from render import render_mesh
from tools import Tools
import numpy as np
import torch

# Define the directory containing the OBJ files


tools = Tools(device="cuda:3")

tools.init_autoencoder(
    "checkpoints/chair-final-bs[128]-lr[0.001]-ks[3]/epoch=987-val_loss=0.0000.ckpt"
)


def downsample_image(image, factor):
    """Downsample an image by the given factor."""
    width, height = image.size
    return image.resize((width // factor, height // factor), Image.Resampling.LANCZOS)


def create_grid_image(grid_size=10, downsample_factor=2):
    images = []

    # Select the first 100 files
    dataset = np.load("encoded_chair_dataset.npy", allow_pickle=True)

    dataset = dataset[: grid_size**2]

    for enc in dataset:
        enc = torch.tensor(enc, device="cuda:3")
        mesh = tools.decode_mesh(enc)
        verts = mesh.verts_list()[0]
        faces = mesh.faces_list()[0]
        verts = verts.cpu().numpy()
        faces = faces.cpu().numpy()
        img = render_mesh(verts, faces)
        img_downsampled = downsample_image(img, downsample_factor)
        images.append(img_downsampled)

    # Determine the size of the grid image
    img_width, img_height = images[0].size
    grid_img_width = img_width * grid_size
    grid_img_height = img_height * grid_size

    # Create a new blank image with the calculated dimensions
    grid_image = Image.new("RGB", (grid_img_width, grid_img_height))

    # Paste each image into the grid
    for i, img in enumerate(images):
        x = (i % grid_size) * img_width
        y = (i // grid_size) * img_height
        grid_image.paste(img, (x, y))

    return grid_image


if __name__ == "__main__":
    grid_image = create_grid_image()
    grid_image.show()  # Display the grid image
    grid_image.save("grid_render.png")  # Optionally save the final grid image
