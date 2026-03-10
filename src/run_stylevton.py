import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import argparse
from torchvision import utils

# Add Flow-Style-VTON/test to path to allow importing models and utils
VTON_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Flow-Style-VTON"))
VTON_TEST_DIR = os.path.join(VTON_DIR, "test")
if VTON_TEST_DIR not in sys.path:
    sys.path.insert(0, VTON_TEST_DIR)

# Reuse the ResUnetGenerator from the repository
from models.networks import ResUnetGenerator, load_checkpoint

# Disable cuDNN to avoid compatibility issues on some Windows systems
torch.backends.cudnn.enabled = False

def prep_rgb(path, device, size=(192, 256)):
    """Prepares RGB image tensor normalized to (-1, 1)."""
    img = Image.open(path).convert('RGB').resize(size, Image.BILINEAR)
    img_np = np.array(img).astype(np.float32)
    img_tensor = (torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0) / 127.5) - 1.0
    return img_tensor.to(device)

def prep_mask(path, device, size=(192, 256)):
    """Prepares mask tensor (binary 0-1)."""
    img = Image.open(path).convert('L').resize(size, Image.NEAREST)
    img_np = np.array(img).astype(np.float32)
    # Binary thresholding
    img_np = (img_np > 127.5).astype(np.float32)
    img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
    return img_tensor.to(device)

def main():
    parser = argparse.ArgumentParser(description="Run Flow-Style-VTON Generation")
    parser.add_argument("--agnostic", required=True, help="Path to agnostic person image")
    parser.add_argument("--original", required=True, help="Path to original person image (no-bg)")
    parser.add_argument("--agnostic_mask", required=True, help="Path to binary agnostic mask")
    parser.add_argument("--warped_cloth", required=True, help="Path to warped garment RGB")
    parser.add_argument("--warped_mask", required=True, help="Path to warped garment mask")
    parser.add_argument("--checkpoint", required=True, help="Path to PFAFN_gen_epoch_101.pth")
    parser.add_argument("--output_path", required=True, help="Path to save the final try-on result")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Initialize Model
    model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d).to(device)
    load_checkpoint(model, args.checkpoint)
    model.eval()
    print("[OK] Generation model loaded.")
    
    # 2. Prepare Tensors
    agnostic_t = prep_rgb(args.agnostic, device)
    original_t = prep_rgb(args.original, device)
    agnostic_mask_t = prep_mask(args.agnostic_mask, device)
    warped_cloth_t = prep_rgb(args.warped_cloth, device)
    warped_mask_t = prep_mask(args.warped_mask, device)
    
    # 3. Generation Inference
    with torch.no_grad():
        # Inputs: Agnostic (3) + Warped Cloth (3) + Warped Mask (1) = 7 channels
        gen_inputs = torch.cat([agnostic_t, warped_cloth_t, warped_mask_t], 1)
        gen_outputs = model(gen_inputs)
        
        # Split outputs: 3 for rendered person, 1 for composition mask
        p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
        p_rendered = torch.tanh(p_rendered)
        m_composite = torch.sigmoid(m_composite)
        
        # Skin inpainting and Composition logic
        m_composite = m_composite * warped_mask_t
        p_tryon = warped_cloth_t * m_composite + p_rendered * (1 - m_composite)
        
        # --- STRATEGIC COMPOSITION ---
        # Instead of just taking the generator's full output (with weird background artifacts),
        # we only use the generator's work inside the AGNOSTIC region.
        # This preserves the original head, original hands/feet, and clean background.
        final_blend = original_t * (1 - agnostic_mask_t) + p_tryon * agnostic_mask_t
        
    # 4. Save Result
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    utils.save_image(
        final_blend,
        args.output_path,
        nrow=1,
        normalize=True,
        range=(-1, 1)
    )
    print(f"[SUCCESS] Final refined try-on saved to: {args.output_path}")

if __name__ == "__main__":
    main()
