import os
import cv2
import numpy as np
import argparse
from PIL import Image

def generate_target_mask(schp_path, densepose_path, output_dir):
    """
    Generates a target mask for virtual try-on by combining 
    human parsing (SCHP) and DensePose data.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load SCHP parsing
    parsing_img = Image.open(schp_path)
    parsing = np.array(parsing_img)
    
    # Load DensePose labels (from visualization or raw output)
    # Background is black (0), so foreground is > 0
    densepose_img = cv2.imread(densepose_path, cv2.IMREAD_GRAYSCALE)
    if densepose_img is None:
        raise ValueError(f"Could not load DensePose image from {densepose_path}")
        
    # Person mask from SCHP (everything that is not background)
    person_mask = (parsing != 0)
    
    # Body mask from DensePose foreground intersect with SCHP person mask
    body_mask = (densepose_img > 0) & person_mask
    
    # Specific parts to EXCLUDE from the target mask (where the garment should NOT go)
    hair_mask = (parsing == 2)
    face_mask = (parsing == 13)
    
    # Lower body/shoes to exclude (indices vary by dataset, using LIP values from notebook)
    # 6: Skirt, 9: Left-shoe, 10: Right-shoe, 12: Pants
    lower_body_mask = np.isin(parsing, [6, 9, 10, 12])
    
    # Start with the body mask and subtract excluded regions
    target_mask = body_mask.copy().astype(np.uint8)
    target_mask[hair_mask] = 0
    target_mask[face_mask] = 0
    target_mask[lower_body_mask] = 0
    
    # Save the target mask as a binary image (0 or 255)
    output_path = os.path.join(output_dir, "target_mask.png")
    cv2.imwrite(output_path, target_mask * 255)
    
    print(f"Target mask generated and saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--schp", required=True, help="Path to SCHP parsing PNG")
    parser.add_argument("--densepose", required=True, help="Path to DensePose output PNG")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    
    generate_target_mask(args.schp, args.densepose, args.output_dir)
