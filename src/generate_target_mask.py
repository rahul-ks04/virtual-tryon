import os
import cv2
import numpy as np
import argparse
from PIL import Image

def generate_target_mask(schp_path, densepose_path, output_dir, sleeve_type="auto", garment_mask=None):
    """
    Generates a target mask for virtual try-on by combining 
    human parsing (SCHP) and DensePose data, with sleeve awareness.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load SCHP parsing
    parsing_img = Image.open(schp_path)
    parsing = np.array(parsing_img)
    
    # Load DensePose labels
    densepose_img = cv2.imread(densepose_path, cv2.IMREAD_GRAYSCALE)
    if densepose_img is None:
        raise ValueError(f"Could not load DensePose image from {densepose_path}")
        
    # Person mask from SCHP
    person_mask = (parsing != 0)
    
    # 1. Sleeve detection if auto
    if sleeve_type == "auto" and garment_mask is not None:
        g_mask = np.array(Image.open(garment_mask).convert('L')) > 127
        h_g, w_g = g_mask.shape
        widths = np.sum(g_mask, axis=1)
        content_rows = np.where(widths > 5)[0]
        if len(content_rows) > 10:
            top_y = content_rows[0]
            bottom_y = content_rows[-1]
            total_h = bottom_y - top_y
            
            # Sample width at 1/3 (shoulders/chest) and 2/3 (forearms level if long sleeve)
            w_top = widths[int(top_y + total_h * 0.3)]
            w_mid = widths[int(top_y + total_h * 0.6)]
            
            # If width drops significantly at mid-length, it's likely short-sleeved
            # Long sleeves usually maintain a wider profile further down the body than short sleeves
            if w_mid < 0.6 * w_top:
                sleeve_type = "short"
            else:
                sleeve_type = "long"
        else:
            sleeve_type = "short"
        print(f"Detected sleeve type for mask generation: {sleeve_type}")

    # 2. Define anatomical regions (LIP Labels)
    # 5: Upper, 6: Dress, 7: Coat
    torso_mask = np.isin(parsing, [5, 6, 7]) 
    arm_mask = np.isin(parsing, [14, 15]) # L-arm, R-arm (skin)
    
    # Base target mask starts with torso/existing clothes
    target_mask_bool = torso_mask.copy()
    
    if sleeve_type == "long":
        # Merge arms to allow warping onto the full length
        target_mask_bool |= arm_mask
    elif sleeve_type == "short":
        # Do not add arms, let the projection handle minor refinements
        pass
    
    # Constraint: Must be within human silhouette as defined by DensePose
    target_mask_bool &= (densepose_img > 0)
    
    target_mask = target_mask_bool.astype(np.uint8)
    
    # Save the target mask as a binary image (0 or 255)
    output_path = os.path.join(output_dir, "target_mask.png")
    cv2.imwrite(output_path, target_mask * 255)
    
    print(f"Target mask generated and saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--schp", required=True, help="Path to SCHP parsing PNG")
    parser.add_argument("--densepose", required=True, help="Path to DensePose output PNG")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sleeve_type", choices=["auto", "short", "long", "sleeveless"], default="auto")
    parser.add_argument("--garment_mask", help="Path to garment mask (required if sleeve_type is auto)")
    args = parser.parse_args()
    
    generate_target_mask(args.schp, args.densepose, args.output_dir, args.sleeve_type, args.garment_mask)
