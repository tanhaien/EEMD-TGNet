import os
from PIL import Image
import numpy as np
import matplotlib.cm as cm

def recolor_monochrome_images(input_dir, output_dir, colormap_name='viridis'):
    """
    Recolors monochromatic images using a specified colormap.
    """
    os.makedirs(output_dir, exist_ok=True)
    colormap = cm.get_cmap(colormap_name)

    print(f"Recoloring images from '{input_dir}' to '{output_dir}' using colormap '{colormap_name}'...")

    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            input_filepath = os.path.join(input_dir, filename)
            output_filepath = os.path.join(output_dir, filename.replace(".png", "_recolored.png"))

            try:
                img_gray = Image.open(input_filepath).convert("L") # Ensure grayscale
                img_array = np.array(img_gray)

                # Normalize pixel values to [0, 1]
                normalized_array = img_array / 255.0

                # Apply colormap
                colored_array = colormap(normalized_array) # This will be RGBA [0,1]

                # Convert to 8-bit RGB
                colored_image = Image.fromarray((colored_array[:, :, :3] * 255).astype(np.uint8))
                
                colored_image.save(output_filepath)
                print(f"  Recolored and saved: {output_filepath}")

            except Exception as e:
                print(f"  Error processing {filename}: {e}")

if __name__ == "__main__":
    input_monochrome_dir = "experiment_results/segmented_images"
    output_recolored_dir = "experiment_results/recolored_images"
    recolor_monochrome_images(input_monochrome_dir, output_recolored_dir)