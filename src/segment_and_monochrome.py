from PIL import Image
import os

def segment_and_monochrome_image(image_path, output_dir):
    """
    Segments a 2x3 subplot image into 6 individual monochromatic sub-images.
    Assumes the input image is a grid of 2 rows and 3 columns of subplots.
    """
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return

    width, height = img.size
    
    # Assuming a 2x3 grid of subplots
    num_rows = 2
    num_cols = 3

    # Calculate dimensions of each subplot
    subplot_width = width // num_cols
    subplot_height = height // num_rows

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing image: {image_path}")
    print(f"Image dimensions: {width}x{height}")
    print(f"Subplot dimensions: {subplot_width}x{subplot_height}")

    for row in range(num_rows):
        for col in range(num_cols):
            left = col * subplot_width
            top = row * subplot_height
            right = left + subplot_width
            bottom = top + subplot_height

            # Crop the subplot
            subplot = img.crop((left, top, right, bottom))

            # Convert to monochromatic (grayscale)
            monochromatic_subplot = subplot.convert("L")

            # Save the monochromatic subplot
            output_filename = f"subplot_row{row+1}_col{col+1}_monochrome.png"
            output_filepath = os.path.join(output_dir, output_filename)
            monochromatic_subplot.save(output_filepath)
            print(f"Saved {output_filepath}")

if __name__ == "__main__":
    input_image_path = "experiment_results/eemd_tgnet_comprehensive_analysis.png"
    output_directory = "experiment_results/segmented_images"
    segment_and_monochrome_image(input_image_path, output_directory)