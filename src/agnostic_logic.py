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
# 1: Hat, 2: Hair, 4: Sunglasses, 13: Face
PRESERVE_LABELS = [1, 2, 4, 13]

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
