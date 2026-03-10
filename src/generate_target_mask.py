import os
import cv2
import numpy as np
import argparse
from PIL import Image

def generate_target_mask(schp_path, densepose_path, output_dir, sleeve_type="full", person_mask_path=None):
    """
    Generates a robust target mask.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Inputs
    parsing = np.array(Image.open(schp_path))
    dp_img = cv2.imread(densepose_path)
    if dp_img is None:
        raise ValueError(f"Could not load DensePose image from {densepose_path}")
    
    dp_i = dp_img[:, :, 0] if len(dp_img.shape) == 3 else dp_img
    
    p_mask = None
    if person_mask_path:
        p_mask_img = Image.open(person_mask_path).convert("L")
        p_mask = (np.array(p_mask_img) > 127).astype(np.uint8)

    # 2. Define Masks
    # Exclude: 1:Hat, 2:Hair, 4:Sunglasses, 13:Face, 16:Pants, 17-20:Legs
    exclude_indices = [1, 2, 4, 13, 16, 17, 18, 19, 20]
    exclude_mask = np.isin(parsing, exclude_indices)
    
    base_torso_garment = np.isin(dp_i, [1, 2]) | np.isin(parsing, [5, 6, 7])
    
    # 3. Refine based on sleeve_type
    if sleeve_type == "full":
        # Strategy: Include Torso + All Arms detection. 
        # FALLBACK: If DensePose/Parsing is weak, but we have a person mask,
        # anything in the person mask that ISN'T excluded (face/pants) and is UPPER body is likely the arm.
        target_mask = base_torso_garment | np.isin(dp_i, range(3, 11)) | np.isin(parsing, [14, 15])
        
        if p_mask is not None:
            # Upper body heuristic: Anything in the person mask that is in the top 75% of the image 
            # and is NOT hat/face/pants.
            h, w = p_mask.shape
            upper_half = np.zeros_like(p_mask, dtype=bool)
            upper_half[:int(h*0.75), :] = True
            
            p_mask_bool = p_mask.astype(bool)
            robust_full_garment = p_mask_bool & upper_half & (~exclude_mask)
            target_mask |= robust_full_garment
            
    elif sleeve_type == "half":
        target_mask = base_torso_garment.copy()
        lower_arms_dp = np.isin(dp_i, [7, 8, 9, 10])
        if np.sum(lower_arms_dp) > 0:
            kernel_la = np.ones((15, 15), np.uint8)
            lower_arms_broad = cv2.dilate(lower_arms_dp.astype(np.uint8), kernel_la, iterations=1).astype(bool)
            target_mask &= (~lower_arms_broad)
            
        y_indices, x_indices = np.where(target_mask)
        if len(y_indices) > 0:
            y_min, y_max = y_indices.min(), y_indices.max()
            x_center = np.median(x_indices)
            x_width = x_indices.max() - x_indices.min()
            elbow_y = y_min + (y_max - y_min) * 0.48
            arm_boundary_dist = x_width * 0.22
            h, w = target_mask.shape
            yy, xx = np.mgrid[:h, :w]
            prune_area = (yy > elbow_y) & (np.abs(xx - x_center) > arm_boundary_dist)
            target_mask &= (~prune_area)
            
        target_mask |= np.isin(dp_i, [3, 4, 5, 6])
    else: # none
        arms_dp = np.isin(dp_i, range(3, 11)) | np.isin(parsing, [14, 15])
        kernel_a = np.ones((25, 25), np.uint8)
        arms_broad = cv2.dilate(arms_dp.astype(np.uint8), kernel_a, iterations=1).astype(bool)
        target_mask = base_torso_garment & (~arms_broad)
    
    # 4. Final Cleanup
    target_mask[exclude_mask] = False
    
    mask_float = target_mask.astype(np.float32)
    kernel = np.ones((7, 7), np.uint8)
    mask_float = cv2.dilate(mask_float, kernel, iterations=1)
    mask_float = cv2.GaussianBlur(mask_float, (15, 15), 5)
    
    # 5. Save Results
    output_path = os.path.join(output_dir, "target_mask.png")
    mask_uint8 = (mask_float * 255).astype(np.uint8)
    
    # 5. Save Results
    output_path = os.path.join(output_dir, "target_mask.png")
    mask_uint8 = (mask_float * 255).astype(np.uint8)
    
    cv2.imwrite(output_path, mask_uint8)
    print(f"Target mask ({sleeve_type}) generated and saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--schp", required=True)
    parser.add_argument("--densepose", required=True)
    parser.add_argument("--person_mask", help="Path to person mask from rembg")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sleeve_type", choices=["none", "half", "full"], default="full")
    args = parser.parse_args()
    
    generate_target_mask(args.schp, args.densepose, args.output_dir, args.sleeve_type, args.person_mask)
