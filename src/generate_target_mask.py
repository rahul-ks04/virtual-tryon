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
    # Keep all masks in the same resolution to avoid stair-step artifacts.
    if dp_i.shape[:2] != parsing.shape[:2]:
        dp_i = cv2.resize(dp_i, (parsing.shape[1], parsing.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    p_mask = None
    if person_mask_path:
        p_mask_img = Image.open(person_mask_path).convert("L")
        p_mask = (np.array(p_mask_img) > 127).astype(np.uint8)
        if p_mask.shape[:2] != parsing.shape[:2]:
            p_mask = cv2.resize(p_mask, (parsing.shape[1], parsing.shape[0]), interpolation=cv2.INTER_NEAREST)

    # 2. Define Masks
    # LIP Labels to EXCLUDE (Preserve):
    # 1: Hat, 2: Hair, 4: Sunglasses, 13: Face (Head)
    # 9: Pants, 12: Skirts/Lower Body, 16: Left-leg, 17: Right-leg, 18: Left-shoe, 19: Right-shoe
    exclude_indices = [1, 2, 4, 9, 12, 13, 16, 17, 18, 19]
    exclude_mask = np.isin(parsing, exclude_indices).astype(np.uint8)
    
    # Dilate exclusion mask slightly to remove "halo" streaks near hair/face
    kernel_ex = np.ones((5, 5), np.uint8)
    exclude_mask = cv2.dilate(exclude_mask, kernel_ex, iterations=1).astype(bool)
    
    # Base area: Torso and anything that looks like upper clothes
    base_torso_garment = np.isin(dp_i, [1, 2]) | np.isin(parsing, [5, 6, 7])
    
    # 3. Refine based on sleeve_type
    if sleeve_type == "full":
        # Strategy: Include Torso + All Arms detection. 
        # FALLBACK: Use person silhouette but strictly block the exclusion zones (Pants/Head)
        target_mask = base_torso_garment | np.isin(dp_i, range(3, 11)) | np.isin(parsing, [14, 15])
        
        if p_mask is not None:
            # Robust garment area = (Person Silhouette) AND NOT (Excluded zone: Pants/Head)
            # This ensures it stops at the waistline naturally.
            p_mask_bool = p_mask.astype(bool)
            robust_full_garment = p_mask_bool & (~exclude_mask)
            target_mask |= robust_full_garment
            
    elif sleeve_type == "half":
        target_mask = base_torso_garment.copy()
        lower_arms_dp = np.isin(dp_i, [7, 8, 9, 10])
        if np.sum(lower_arms_dp) > 0:
            # Smaller elliptical kernel reduces chunky sleeve cutoffs.
            kernel_la = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
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

        # Smooth short-sleeve boundaries so they are less blocky.
        target_u8 = target_mask.astype(np.uint8)
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        target_u8 = cv2.morphologyEx(target_u8, cv2.MORPH_CLOSE, k_close)
        target_u8 = cv2.morphologyEx(target_u8, cv2.MORPH_OPEN, k_open)
        target_u8 = cv2.medianBlur((target_u8 * 255).astype(np.uint8), 5)
        target_mask = (target_u8 > 127)
    else: # none
        arms_dp = np.isin(dp_i, range(3, 11)) | np.isin(parsing, [14, 15])
        kernel_a = np.ones((25, 25), np.uint8)
        arms_broad = cv2.dilate(arms_dp.astype(np.uint8), kernel_a, iterations=1).astype(bool)
        target_mask = base_torso_garment & (~arms_broad)
    
    # 4. Final Cleanup (Ensure no pants/head pixels slipped in)
    target_mask[exclude_mask] = False
    
    mask_float = target_mask.astype(np.float32)
    # Final universal refinement with smooth edges.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_float = cv2.dilate(mask_float, kernel, iterations=1)
    mask_float = cv2.GaussianBlur(mask_float, (11, 11), 2)
    
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
