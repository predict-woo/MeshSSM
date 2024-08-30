import os
from PIL import Image
import igl
from render import render_mesh

# Define the directory containing the OBJ files
directory = "/home/andyye/MeshSSM/chair_dataset"


def downsample_image(image, factor):
    """Downsample an image by the given factor."""
    width, height = image.size
    return image.resize((width // factor, height // factor), Image.Resampling.LANCZOS)


def create_grid_image(directory, grid_size=10, downsample_factor=2):
    images = []

    # Get all .obj files and sort them numerically
    files = sorted(
        [f for f in os.listdir(directory) if f.endswith(".obj")],
        key=lambda x: int(x.split(".")[0]),
    )

    # Select the first 100 files
    files = files[: grid_size**2]

    for filename in files:
        obj_path = os.path.join(directory, filename)
        vertices, faces = igl.read_triangle_mesh(obj_path)
        img = render_mesh(vertices, faces)
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
    grid_image = create_grid_image(directory)
    grid_image.show()  # Display the grid image
    grid_image.save("grid_render.png")  # Optionally save the final grid image
