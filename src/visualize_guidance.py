
import os
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))
from agnostic_logic import get_densepose_guidance, PRESERVE_LABELS

def create_mock_densepose(parse_path, output_path):
    """
    Creates a mock DensePose map based on the parsing map.
    We'll make the arms slightly longer/thicker to simulate sleeveless-to-sleeved support.
    """
    parse = np.array(Image.open(parse_path).convert("L"))
    
    # Identify torso and arms from LIP parsing
    # 5: Upper-clothes, 14: L-arm, 15: R-arm
    body_mask = np.isin(parse, [5, 14, 15]).astype(np.uint8)
    
    # Expand it to simulate 'body surface' coverage
    kernel = np.ones((7,7), np.uint8)
    mock_dp = cv2.dilate(body_mask, kernel, iterations=3)
    
    # Convert to typical DensePose labels (1 for torso, 11 for arms)
    # This is a simplification but enough for the logic check
    mock_dp_labeled = np.zeros_like(mock_dp)
    mock_dp_labeled[mock_dp == 1] = 1 # Torso
    
    Image.fromarray(mock_dp_labeled).save(output_path)
    print(f"Mock DensePose saved to {output_path}")

def main():
    # Use existing parsing map
    parse_path = Path(r"d:\Virtual try on\virtual-tryon\outputs\schp\00005_00.png")
    if not parse_path.exists():
        print(f"Error: Could not find parse at {parse_path}")
        return

    # Create dummy densepose path
    dp_path = parse_path.parent / "00005_00_dp_mock.png"
    create_mock_densepose(parse_path, dp_path)

    # Run guidance generation
    guidance = get_densepose_guidance(parse_path, dp_path)
    guidance_arr = np.array(guidance)

    # Visualization
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    ax[0].imshow(np.array(Image.open(parse_path)), cmap='nipy_spectral')
    ax[0].set_title("Original Parsing Map")
    
    ax[1].imshow(guidance_arr, cmap='nipy_spectral')
    ax[1].set_title("Target Guidance Map (FEM)")
    
    # Check for label 5 (Upper clothes) distribution
    label_5_mask = (guidance_arr == 5)
    print(f"Label 5 pixels in guidance: {np.sum(label_5_mask)}")
    
    # Check if preservation regions are intact
    preserve_mask = np.isin(guidance_arr, PRESERVE_LABELS)
    print(f"Preservation label pixels: {np.sum(preserve_mask)}")

    output_viz = Path(r"C:\Users\rahul\.gemini\antigravity\brain\282ed171-e9d1-4433-ab2c-40e530fddf37\target_guidance_viz.png")
    plt.savefig(output_viz)
    print(f"Visualization saved to {output_viz}")

if __name__ == "__main__":
    main()
