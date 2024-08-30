import os
import re
from PIL import Image, ImageDraw, ImageFont
import igl
from render import render_mesh

# Define the directory containing the OBJ files
directory = "results/chair-final_ks[3]_lr[0.001]"


def numerical_sort(value):
    # Extract numbers from the filename for numerical sorting
    numbers = re.findall(r"\d+", value)
    return int(numbers[0]) if numbers else -1


def add_title_to_image(image, step):
    # Create a new image with space for the title
    width, height = image.size
    title_height = 200  # Height for the title area
    new_height = height + title_height  # Add more space at the top for the larger title
    new_image = Image.new("RGB", (width, new_height), "white")

    # Paste the original image onto the new image
    new_image.paste(image, (0, title_height))

    # Draw the title on the new image
    draw = ImageDraw.Draw(new_image)
    title_text = f"Step: {step}"

    # Load a TrueType font with a larger size
    font = ImageFont.truetype(
        "/home/andyye/MeshSSM/project/SpoqaHanSansNeo-Medium.ttf", 100
    )

    # Calculate the text size using textbbox
    text_bbox = draw.textbbox((0, 0), title_text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    text_x = (width - text_width) // 2  # Center the title horizontally
    text_y = (title_height - text_height) // 2  # Vertically center in the title space

    draw.text((text_x, text_y), title_text, fill="black", font=font)

    return new_image


def stitch_images(images):
    # Calculate the total width and maximum height of the final image
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)

    # Create a new blank image with the calculated dimensions
    stitched_image = Image.new("RGB", (total_width, max_height))

    # Paste each image next to the previous one
    current_x = 0
    for img in images:
        stitched_image.paste(img, (current_x, 0))
        current_x += img.width

    return stitched_image


def render_and_stitch(directory):
    images = []
    files = sorted(os.listdir(directory), key=numerical_sort)

    # Iterate over every 20th file in the directory
    for i, filename in enumerate(files):

        if i < 8 * 20:
            continue

        if i >= 12 * 20:
            break

        if i % 20 == 0 and filename.endswith(".obj"):
            obj_path = os.path.join(directory, filename)
            step = numerical_sort(filename)  # Extract the step from the filename
            vertices, faces = igl.read_triangle_mesh(obj_path)
            img = render_mesh(vertices, faces)
            img_with_title = add_title_to_image(img, step)  # Add title to the image
            images.append(img_with_title)

    # Stitch the images together horizontally
    if images:
        stitched_image = stitch_images(images)
        return stitched_image
    else:
        print("No images found to stitch.")
        return None


if __name__ == "__main__":
    stitched_image = render_and_stitch(directory)
    if stitched_image:
        stitched_image.show()  # Display the stitched image
        stitched_image.save(
            "stitched_render.png"
        )  # Optionally save the final stitched image
