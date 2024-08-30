import pyvista as pv
from pytorch3d.io import load_obj
import numpy as np
from PIL import Image

# Start virtual framebuffer
pv.start_xvfb()


def render_mesh(vertices, faces):
    # Load the OBJ file
    # Load the OBJ file using igl

    # PyVista expects faces to be prefixed with the number of vertices per face (usually 3 for triangles)
    faces = np.hstack([np.full((faces.shape[0], 1), 3), faces])

    # Create a PyVista mesh
    pv_mesh = pv.PolyData(vertices, faces)

    # Create a Plotter object with increased resolution (e.g., 1920x1080)
    plotter = pv.Plotter(off_screen=True, window_size=(1024, 1024))  # 1080p resolution

    # Add the mesh to the plotter with filled faces
    plotter.add_mesh(
        pv_mesh,
        color="#FABC3F",
        # opacity=0.6,
        show_edges=True,
        edge_color="black",
        line_width=2,
    )

    # Set the background color
    plotter.set_background("white")

    # Set the camera position manually
    camera_position = [
        (2, 2, -2),  # Camera position (x, y, z)
        (0.5, 0.5, 0.5),  # Focal point (center of the object)
        (0, 1, 0),  # View up vector (defines the upward direction)
    ]
    plotter.camera_position = camera_position

    # Render the scene and save the image with the fixed name 'render.png'
    img_array = plotter.screenshot(return_img=True)

    img = Image.fromarray(img_array)
    return img


if __name__ == "__main__":
    import igl

    vertices, faces = igl.read_triangle_mesh(
        "results/chair-final_ks[3]_lr[0.001]/mesh_15984.obj"
    )
    render_mesh(vertices, faces)
