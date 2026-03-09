import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import argparse

# LIP (Look Into Person) Labels Mapping for Upper Body Agnostic
# Labels to MASK OUT (replace with neutral gray):
# 5: Upper-clothes, 6: Dress, 7: Coat, 10: Jumpsuits, 11: Scarf, 14: Left-arm, 15: Right-arm
AGNOSTIC_LABELS = [5, 6, 7, 10, 11, 14, 15]

# Labels to PRESERVE (keep original pixels even if mask overlaps):
# 1: Hat, 2: Hair, 4: Sunglasses, 13: Face
PRESERVE_LABELS = [1, 2, 4, 13]

def generate_agnostic(img_path, parse_path, output_dir, dilation_kernel_size=25, smoothing_sigma=5, preserve_arms=False):
    """
    Generates agnostic image and parsing map.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load image and parsing
    image = np.array(Image.open(img_path).convert("RGB"))
    parse = np.array(Image.open(parse_path))

    # 1. Create preservation mask (Face, hair, etc.)
    current_preserve_labels = PRESERVE_LABELS.copy()
    if preserve_arms:
        current_preserve_labels += [14, 15]
    
    preserve_mask = np.isin(parse, current_preserve_labels).astype(np.uint8)

    # 2. Create binary mask for agnostic areas (Clothes and potentially Arms)
    current_agnostic_labels = [5, 6, 7, 10, 11] # Clothes
    if not preserve_arms:
        current_agnostic_labels += [14, 15] # Add arms if not preserving
    
    agnostic_mask = np.isin(parse, current_agnostic_labels).astype(np.uint8)

    # 3. Dilate the agnostic mask to break 'Cloth Shape Bias'
    kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    dilated_mask = cv2.dilate(agnostic_mask, kernel, iterations=1)

    # 4. Refine Mask: Ensure preservation areas are NOT covered
    # (e.g. don't let gray area cover the face or preserved arms)
    final_agnostic_mask = (dilated_mask == 1) & (preserve_mask == 0)

    # 5. Optional: Smooth the mask edges to avoid harsh transitions
    mask_float = final_agnostic_mask.astype(float)
    if smoothing_sigma > 0:
        mask_float = cv2.GaussianBlur(mask_float, (0, 0), smoothing_sigma)

    # Create Agnostic Parsing
    # We clear the garment/arm labels from the parsing map and set them to background (0)
    agnostic_parse = parse.copy()
    agnostic_parse[np.isin(parse, current_agnostic_labels)] = 0

    # Create Agnostic Image
    gray_img = np.full_like(image, 128, dtype=np.uint8)
    mask_float_3d = mask_float[:, :, None]
    
    # Blending
    agnostic_img_float = (1.0 - mask_float_3d) * image.astype(float) + mask_float_3d * gray_img.astype(float)
    agnostic_img = np.clip(agnostic_img_float, 0, 255).astype(np.uint8)

    # 6. Create Color Debug Visualization
    # Overlay the mask in Red on the original image to see what's being covered
    debug_img = image.copy()
    red_overlay = np.zeros_like(image)
    red_overlay[:,:,0] = 255 # Only red channel
    
    # debug = image * (1 - mask) + red_overlay * mask
    debug_float = (1.0 - mask_float_3d) * image.astype(float) + mask_float_3d * red_overlay.astype(float)
    debug_img = np.clip(debug_float, 0, 255).astype(np.uint8)

    # Save results
    img_name = Path(img_path).name
    Image.fromarray(agnostic_img).save(os.path.join(output_dir, img_name))
    
    # Save debug image
    debug_name = "debug_" + img_name
    Image.fromarray(debug_img).save(os.path.join(output_dir, debug_name))
    
    # Save parsing as PNG
    parse_name = Path(parse_path).name
    Image.fromarray(agnostic_parse.astype(np.uint8)).save(os.path.join(output_dir, parse_name))

    print(f"Generated agnostic person: {os.path.join(output_dir, img_name)}")
    print(f"Generated debug visualization: {os.path.join(output_dir, debug_name)}")
    print(f"Input image shape: {image.shape}, Output image shape: {agnostic_img.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Agnostic Person for VITON")
    parser.add_argument("--image", required=True, help="Path to original person image")
    parser.add_argument("--parse", required=True, help="Path to parsing map (.png)")
    parser.add_argument("--output_dir", required=True, help="Directory to save output")
    parser.add_argument("--dilation", type=int, default=25, help="Kernel size for mask dilation")
    parser.add_argument("--smoothing", type=int, default=5, help="Sigma for Gaussian smoothing of the mask")
    parser.add_argument("--preserve_arms", action="store_true", help="Do not mask arms (useful for crossed arms)")

    args = parser.parse_args()
    generate_agnostic(args.image, args.parse, args.output_dir, args.dilation, args.smoothing, args.preserve_arms)
