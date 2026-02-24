import cv2
import numpy as np
from PIL import Image
import os

AGNOSTIC_LABELS = [5, 6, 7, 14, 15]

def get_agnostic_person(img_path, parse_path, dilation_kernel_size=25):
    image = np.array(Image.open(img_path).convert("RGB"))
    # The uploaded parse image might be color-coded (RGB) or indices.
    # In VITON-HD dataset, image-parse is often color-coded, but LIP labels are indices.
    # However, for this verification, we assume the parse map values match LIP indices.
    # If the image is color-encoded, we'd need a mapping back to indices.
    # Looking at the schp.ipynb logic, it outputs class indices.
    parse = np.array(Image.open(parse_path))
    
    # If parse is 3D (RGB), we might need to handle it. Usually, it's 2D indices.
    if len(parse.shape) == 3:
        # Simplistic approach to handle potential RGB parse images by checking color triggers
        # or assuming the first channel if it's not a proper index map.
        # But in many datasets, 'image-parse' refers to the colorized version.
        # Let's assume it's a standard index map for now.
        parse = parse[:,:,0] 

    mask = np.isin(parse, AGNOSTIC_LABELS).astype(np.uint8)
    kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)

    agnostic_img = image.copy()
    agnostic_img[dilated_mask == 1] = [128, 128, 128]
    
    agnostic_parse = parse.copy()
    agnostic_parse[mask == 1] = 0

    return Image.fromarray(agnostic_img), Image.fromarray(agnostic_parse)

# Paths for uploaded images
img_path = "C:/Users/rahul/.gemini/antigravity/brain/7b086bf5-ca1d-4004-bf03-9f023c1a0297/uploaded_media_0_1771924789261.png"
parse_path = "C:/Users/rahul/.gemini/antigravity/brain/7b086bf5-ca1d-4004-bf03-9f023c1a0297/uploaded_media_1_1771924789261.png"
output_dir = "C:/Users/rahul/.gemini/antigravity/brain/7b086bf5-ca1d-4004-bf03-9f023c1a0297"

agn_img, agn_prse = get_agnostic_person(img_path, parse_path)
agn_img.save(os.path.join(output_dir, "test_agnostic_person.png"))
agn_prse.save(os.path.join(output_dir, "test_agnostic_parse.png"))
print("Verification images saved.")
