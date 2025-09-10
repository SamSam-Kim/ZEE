import os
import cv2
from pathlib import Path

def create_meta_info_file():
    """
    Scans a directory for images and creates a meta_info.txt file.
    The format is: "image_name.png (height, width, channels)"
    """
    # --- 1. USER CONFIGURATION ---
    # The folder containing your ground-truth training images
    image_folder = 'datasets/sample'
    
    # The name of the output meta info file
    output_filename = 'meta_info_my_custom_data_GT.txt'
    # -----------------------------

    # Create the directory for the meta info file if it doesn't exist
    output_dir = Path('hat/data/meta_info')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename

    if not os.path.isdir(image_folder):
        print(f"Error: Image folder not found at '{image_folder}'\n" \
              "Please create it and place your training images inside.")
        return

    image_paths = sorted([p for p in Path(image_folder).glob('*') if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']])

    if not image_paths:
        print(f"Error: No images found in '{image_folder}'.")
        return

    print(f"Found {len(image_paths)} images in '{image_folder}'.")

    with open(output_path, 'w') as f:
        for img_path in image_paths:
            try:
                # Read image to get its dimensions
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"Warning: Could not read image {img_path}, skipping.")
                    continue
                h, w, c = img.shape
                
                # Write the meta info line
                # Format: 0001.png (1080, 1920, 3)
                line = f"{img_path.name} ({h}, {w}, {c})\n"
                f.write(line)
            except Exception as e:
                print(f"Warning: Error processing image {img_path}: {e}, skipping.")

    print(f"Successfully created meta info file at: {output_path}")

if __name__ == '__main__':
    create_meta_info_file()
