import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

# LIP Labels Mapping
# Labels to MASK OUT (Gray area):
# 5: Upper-clothes, 6: Dress, 7: Coat, 14: Left-arm, 15: Right-arm
AGNOSTIC_LABELS = [5, 6, 7, 14, 15]

# Labels to PRESERVE (Keep original pixels even if mask overlaps):
# 1: Hat, 2: Hair, 3: Glove, 4: Sunglasses, 10: Neck, 13: Face
PRESERVE_LABELS = [1, 2, 3, 4, 10, 13]

def get_agnostic_person(img_path, parse_path, dilation_kernel_size=25):
    """
    Generates agnostic image and parsing map with feature preservation.
    """
    # Load image and parsing
    image = np.array(Image.open(Path(img_path)).convert("RGB"))
    parse = np.array(Image.open(Path(parse_path)))

    # 1. Create preservation mask (Parts of the person we want to keep at all costs)
    preserve_mask = np.isin(parse, PRESERVE_LABELS).astype(np.uint8)

    # 2. Create binary mask for agnostic areas (Clothes and Arms)
    agnostic_mask = np.isin(parse, AGNOSTIC_LABELS).astype(np.uint8)

    # 3. Dilate the agnostic mask to break 'Cloth Shape Bias'
    kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    dilated_mask = cv2.dilate(agnostic_mask, kernel, iterations=1)

    # 4. Refine Dilated Mask: Subtract the preservation mask
    # This ensures gray area doesn't cover hair, face, or hat
    final_agnostic_mask = (dilated_mask == 1) & (preserve_mask == 0)

    # Create Agnostic Image
    agnostic_img = image.copy()
    # Replace masked area with neutral gray (128, 128, 128)
    agnostic_img[final_agnostic_mask] = [128, 128, 128]
    
    # Create Agnostic Parsing
    # We clear the garment/arm labels from the parsing map
    agnostic_parse = parse.copy()
    agnostic_parse[agnostic_mask == 1] = 0 # Set to background

    return Image.fromarray(agnostic_img), Image.fromarray(agnostic_parse)

def get_densepose_guidance(parse_path, densepose_path):
    """
    Generates a neutral 'Guidance Map' for the Flow Estimation Module (FEM).
    
    Instead of using the original garment's silhouette (which may be a 
    sleeveless top), we use DensePose body segmentation to identify the 
    actual surface of the torso and arms.
    
    Target Logic:
    1. Extract torso and arm regions from DensePose.
    2. Apply morphological operations to fill gaps (like folded arms) 
       and smooth the target area.
    3. Ensure preservation labels (face, hair) are NOT overwritten.
    4. Fill the resulting neutral region with Label 5 (Upper Clothes).
    """
    parse = np.array(Image.open(Path(parse_path)).convert("L"))
    dp_segm = np.array(Image.open(Path(densepose_path)).convert("L"))

    # DensePose Labels: 
    # 1, 2 = Torso (Ideal canvas for shirts/dresses)
    # 11, 13 = Left Arm regions
    # 12, 14 = Right Arm regions
    guidance_mask = np.isin(dp_segm, [1, 2, 11, 12, 13, 14]).astype(np.uint8)

    # Refinement: Fill holes and smooth edges to create a 'neutral' canvas
    kernel = np.ones((5,5), np.uint8)
    # Closing handles folded arms (fills small gaps between arm and torso)
    neutral_target = cv2.morphologyEx(guidance_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    # Dilation ensures the flow has a 'safe' target padding
    neutral_target = cv2.dilate(neutral_target, kernel, iterations=2)

    # PROTECTION: Do not allow the guidance mask to overwrite face/hair/hat
    preserve_mask = np.isin(parse, PRESERVE_LABELS).astype(np.uint8)
    neutral_target[preserve_mask == 1] = 0

    # Construct Guidance Flow Map
    guidance = parse.copy()
    
    # 1. Clear original garment labels (5=Upper, 6=Dress, 7=Coat) 
    # to avoid shape leak
    guidance[np.isin(parse, [5, 6, 7])] = 0
    
    # 2. Assign neutral body surface as new 'target' area (Label 5)
    guidance[neutral_target == 1] = 5

    return Image.fromarray(guidance)
