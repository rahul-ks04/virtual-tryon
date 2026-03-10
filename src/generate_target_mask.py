import os
import cv2
import numpy as np
import argparse
from PIL import Image

def generate_target_mask(schp_path, densepose_path, output_dir, sleeve_type="full"):
    """
    Generates a robust target mask based on the approved strategy.
    
    Args:
        schp_path: Path to SCHP parsing image.
        densepose_path: Path to DensePose IUV image.
        output_dir: Directory to save the output.
        sleeve_type: 'none', 'half', or 'full'.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Inputs
    parsing = np.array(Image.open(schp_path))
    dp_img = cv2.imread(densepose_path)
    if dp_img is None:
        raise ValueError(f"Could not load DensePose image from {densepose_path}")
    
    # DensePose Index 'I'
    dp_i = dp_img[:, :, 0] if len(dp_img.shape) == 3 else dp_img

    # 2. Define Masks
    # Preservation (Exclude from target mask)
    # 1: Hat, 2: Hair, 4: Sunglasses, 13: Face
    # 9: Pants, 12: Skirts, 16: L-Leg, 17: R-Leg, 18: L-Shoe, 19: R-Shoe
    exclude_indices = [1, 2, 4, 9, 12, 13, 16, 17, 18, 19]
    exclude_mask = np.isin(parsing, exclude_indices)
    
    # Core Torso (Always include)
    # DensePose 1, 2: Torso
    # SCHP 5: UpperClothes, 6: Dress, 7: Coat
    target_mask = np.isin(dp_i, [1, 2]) | np.isin(parsing, [5, 6, 7])
    
    # 3. Handle Arms based on sleeve_type
    # DensePose 3-6: Upper Arms, 7-10: Lower Arms
    # SCHP 14, 15: Arms
    if sleeve_type == "full":
        target_mask |= np.isin(dp_i, range(3, 11)) | np.isin(parsing, [14, 15])
    elif sleeve_type == "half":
        target_mask |= np.isin(dp_i, [3, 4, 5, 6])
    
    # 4. Final Cleanup
    # Ensure no preservation areas are in the target mask
    target_mask[exclude_mask] = False
    
    # Convert to float for processing
    mask_float = target_mask.astype(np.float32)
    
    # Dilation to bridge gaps
    kernel = np.ones((7, 7), np.uint8)
    mask_float = cv2.dilate(mask_float, kernel, iterations=1)
    
    # Gaussian Blur for smooth edges (essential for flow renderer)
    mask_float = cv2.GaussianBlur(mask_float, (15, 15), 5)
    
    # 5. Save Results
    output_path = os.path.join(output_dir, "target_mask.png")
    cv2.imwrite(output_path, (mask_float * 255).astype(np.uint8))
    
    print(f"Target mask ({sleeve_type}) generated and saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--schp", required=True, help="Path to SCHP parsing PNG")
    parser.add_argument("--densepose", required=True, help="Path to DensePose output PNG")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sleeve_type", choices=["none", "half", "full"], default="full")
    args = parser.parse_args()
    
    generate_target_mask(args.schp, args.densepose, args.output_dir, args.sleeve_type)
