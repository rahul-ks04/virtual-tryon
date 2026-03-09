import os
import numpy as np
from PIL import Image
import argparse
from pathlib import Path

def compose_tryon(agnostic_path, warped_garment_path, warped_mask_path, output_dir):
    """
    Blends the warped garment onto the agnostic person image using the mask.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load images
    agnostic = np.array(Image.open(agnostic_path).convert("RGB"))
    warped_garment = np.array(Image.open(warped_garment_path).convert("RGB"))
    
    # Load mask and normalize to 0-1
    mask = Image.open(warped_mask_path).convert("L")
    mask_np = np.array(mask).astype(float) / 255.0

    # Ensure all have same dimensions
    if agnostic.shape[:2] != warped_garment.shape[:2]:
        print("Warning: Agnostic and warped garment have different dimensions. Resizing garment...")
        h, w = agnostic.shape[:2]
        warped_garment = np.array(Image.fromarray(warped_garment).resize((w, h)))
        mask_np = np.array(mask.resize((w, h))).astype(float) / 255.0

    # Expand mask for multiplication
    mask_np = mask_np[:, :, None]

    # Blend: result = (1 - mask) * agnostic + mask * warped_garment
    # If mask is 1, take warped garment. If 0, take agnostic.
    result = (1.0 - mask_np) * agnostic + mask_np * warped_garment
    result = result.astype(np.uint8)

    # Save output
    output_name = Path(agnostic_path).name
    save_path = os.path.join(output_dir, output_name)
    Image.fromarray(result).save(save_path)
    
    print(f"Final try-on result saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compose Final Virtual Try-On Result")
    parser.add_argument("--agnostic", required=True, help="Path to agnostic person image")
    parser.add_argument("--warped_garment", required=True, help="Path to warped garment image")
    parser.add_argument("--warped_mask", required=True, help="Path to warped garment mask")
    parser.add_argument("--output_dir", required=True, help="Output directory")

    args = parser.parse_args()
    compose_tryon(args.agnostic, args.warped_garment, args.warped_mask, args.output_dir)
