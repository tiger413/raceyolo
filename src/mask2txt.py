import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def mask_to_yolo_single(mask_path, output_path, threshold=128):
    """
    Convert a single binary mask to YOLO format annotation.
    Handles cases where the mask is empty (pure background).

    Args:
        mask_path (str): Path to the binary mask
        output_path (str): Path to save the YOLO format annotation
        threshold (int): Threshold for binary mask, pixels above this value are considered foreground

    Returns:
        bool: True if conversion was successful, False otherwise
    """
    try:
        # Load mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise IOError(f"Unable to read mask at {mask_path}")

        # Threshold mask if needed
        _, mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)

        # Check if mask has any foreground pixels
        if np.sum(mask) == 0:
            # Create empty txt file for background-only images
            with open(output_path, 'w') as f:
                pass
            return True

        # Get mask dimensions
        height, width = mask.shape
        # Calculate dynamic minimum area (0.001% of image area)
        min_area = 0.00001 * height * width

        # Find contours in the mask - changed to RETR_LIST to get all contours
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Open file to write YOLO format annotations
        with open(output_path, 'w') as f:
            has_valid_contours = False
            for contour in contours:
                # Filter out very small contours
                if cv2.contourArea(contour) < min_area:
                    continue

                has_valid_contours = True
                # Get bounding box coordinates
                x, y, w, h = cv2.boundingRect(contour)

                # Calculate normalized coordinates for center point and dimensions
                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height
                w_normalized = w / width
                h_normalized = h / height

                # Ensure values are within valid range [0, 1]
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                w_normalized = max(0, min(1, w_normalized))
                h_normalized = max(0, min(1, h_normalized))

                # Write to file in YOLO format: <class> <x_center> <y_center> <width> <height>
                f.write(f"0 {x_center:.6f} {y_center:.6f} {w_normalized:.6f} {h_normalized:.6f}\n")

            # If no valid contours were found (all were too small), treat as background
            if not has_valid_contours:
                # Ensure file is empty
                f.seek(0)
                f.truncate()

        return True

    except Exception as e:
        print(f"Error processing {mask_path}: {str(e)}")
        return False


def process_folder(mask_folder, output_folder, file_types=('.png', '.jpg', '.jpeg')):
    """
    Process all masks in the given folder and convert to YOLO format.
    Creates empty txt files for background-only images.

    Args:
        mask_folder (str): Path to folder containing binary masks
        output_folder (str): Path to save YOLO format annotations
        file_types (tuple): Tuple of allowed mask file extensions
    """
    # Convert to Path objects
    mask_folder = Path(mask_folder)
    output_folder = Path(output_folder)

    # Create output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)

    # Get list of mask files
    mask_files = [f for f in mask_folder.iterdir() if f.suffix.lower() in file_types]

    # Process masks with progress bar
    with tqdm(total=len(mask_files), desc="Converting masks") as pbar:
        for mask_path in mask_files:
            # Generate output path for labels
            output_path = output_folder / f"{mask_path.stem}.txt"

            # Convert mask to YOLO format
            success = mask_to_yolo_single(mask_path, output_path)

            if success:
                pbar.set_postfix(file=mask_path.name)
            pbar.update(1)


if __name__ == "__main__":
    # Example usage
    mask_folder = "/NUDT-SIRST/masks"
    output_folder = "/NUDT-SIRST/labels"

    process_folder(mask_folder, output_folder)

