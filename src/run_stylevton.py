import os
import cv2
import numpy as np
from PIL import Image
import argparse

# ---------------------------------------------------------------------------
# LIP / SCHP label groups
# ---------------------------------------------------------------------------
# Preservation zone — these pixels are NEVER touched, always taken from original
PRESERVE_LABELS  = [1, 2, 4, 13]   # Hat, Hair, Sunglasses, Face
# Old garment region — erase before pasting new cloth
GARMENT_LABELS   = [5, 6, 7, 10, 11]  # Upper-clothes, Dress, Coat, Jumpsuit, Scarf
# Arm/hand labels — erased only where new garment covers them
ARM_LABELS       = [14, 15]         # Left-arm, Right-arm
# Lower body — never touched
LOWER_LABELS     = [9, 12, 16, 17, 18, 19]  # Pants, Skirt, Left-leg, Right-leg, Left-shoe, Right-shoe


def load_rgba(path, target_size=None):
    """Load image as float32 RGBA [0,1]. target_size = (W, H)."""
    img = Image.open(path).convert("RGBA")
    if target_size:
        img = img.resize(target_size, Image.LANCZOS)
    return np.array(img).astype(np.float32) / 255.0


def load_rgb(path, target_size=None):
    """Load image as float32 RGB [0,1]. target_size = (W, H)."""
    img = Image.open(path).convert("RGB")
    if target_size:
        img = img.resize(target_size, Image.LANCZOS)
    return np.array(img).astype(np.float32) / 255.0


def load_mask(path, target_size=None, threshold=0.5):
    """Load image as binary float32 mask [0 or 1]. target_size = (W, H)."""
    img = Image.open(path).convert("L")
    if target_size:
        img = img.resize(target_size, Image.NEAREST)
    arr = np.array(img).astype(np.float32) / 255.0
    return (arr > threshold).astype(np.float32)


def load_parse(path, target_size=None):
    """Load SCHP parsing map as uint8. target_size = (W, H)."""
    img = Image.open(path)
    if target_size:
        img = img.resize(target_size, Image.NEAREST)
    return np.array(img).astype(np.uint8)


def soft_mask(binary_mask, blur_radius=7):
    """
    Feather a binary mask with a Gaussian blur for smooth blending edges.
    blur_radius must be odd.
    """
    r = blur_radius | 1  # ensure odd
    return cv2.GaussianBlur(binary_mask.astype(np.float32), (r, r), 0)


def build_label_mask(parse, labels):
    """Return binary mask where any of `labels` is present in `parse`."""
    mask = np.zeros(parse.shape, dtype=np.uint8)
    for lbl in labels:
        mask |= (parse == lbl).astype(np.uint8)
    return mask.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Layered Virtual Try-On Compositor (no GAN inpainting)"
    )
    # --- inputs shared with old interface so master_pipeline.py needs no changes ---
    parser.add_argument("--agnostic",      required=True,  help="Path to agnostic person image (kept for compatibility, not used for synthesis)")
    parser.add_argument("--original",      required=True,  help="Path to original person image WITHOUT background (rembg RGBA .png)")
    parser.add_argument("--agnostic_mask", required=True,  help="Path to binary agnostic mask (kept for compatibility)")
    parser.add_argument("--warped_cloth",  required=True,  help="Path to warped garment RGB image")
    parser.add_argument("--warped_mask",   required=True,  help="Path to warped garment binary mask")
    parser.add_argument("--parse",         default=None,   help="Path to SCHP parsing map of the person (strongly recommended)")
    parser.add_argument("--output_path",   required=True,  help="Path to save the final try-on result (.png)")
    # --- compositing controls ---
    parser.add_argument("--feather",       type=int, default=21,
                        help="Gaussian blur radius (px) for garment mask feathering. Larger = softer edge.")
    parser.add_argument("--garment_dilation", type=int, default=5,
                        help="Morphological dilation (px) applied to old-garment erase mask before erasing.")
    parser.add_argument("--preserve_feather", type=int, default=9,
                        help="Feather radius for the hard-preserve zone (face/hair) copy-back.")
    # kept for master_pipeline CLI compatibility
    parser.add_argument("--checkpoint",    default=None,   help="(Unused) legacy checkpoint arg")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 0. Resolve working size from the original person image
    # ------------------------------------------------------------------
    orig_pil = Image.open(args.original).convert("RGBA")
    W, H = orig_pil.size
    sz = (W, H)
    print(f"[INFO] Working resolution: {W}x{H}")

    # ------------------------------------------------------------------
    # 1. Load all inputs at working resolution
    # ------------------------------------------------------------------
    person_rgba = load_rgba(args.original, sz)          # (H,W,4) — rembg RGBA
    person_rgb  = person_rgba[:, :, :3]                  # (H,W,3)
    person_alpha = person_rgba[:, :, 3:4]                # (H,W,1) — person silhouette

    warped_cloth = load_rgb(args.warped_cloth, sz)       # (H,W,3)
    warped_mask  = load_mask(args.warped_mask, sz)       # (H,W)   binary

    # SCHP parsing — optional but strongly recommended
    if args.parse and os.path.exists(args.parse):
        parse = load_parse(args.parse, sz)               # (H,W)   uint8 label ids
        has_parse = True
        print("[INFO] SCHP parsing loaded.")
    else:
        parse = None
        has_parse = False
        print("[WARN] No SCHP parse provided — arm/garment erasing will be approximate.")

    # ------------------------------------------------------------------
    # 2. Build semantic masks from SCHP
    # ------------------------------------------------------------------
    if has_parse:
        old_garment_mask = build_label_mask(parse, GARMENT_LABELS)   # torso clothing
        arm_mask         = build_label_mask(parse, ARM_LABELS)        # both arms
        preserve_mask    = build_label_mask(parse, PRESERVE_LABELS)   # face/hair — NEVER touch
        lower_mask       = build_label_mask(parse, LOWER_LABELS)      # pants/shoes — NEVER touch
    else:
        # Fallback: use agnostic mask as erase region, no arm/preserve logic
        old_garment_mask = load_mask(args.agnostic_mask, sz)
        arm_mask         = np.zeros((H, W), dtype=np.float32)
        preserve_mask    = np.zeros((H, W), dtype=np.float32)
        lower_mask       = np.zeros((H, W), dtype=np.float32)

    # ------------------------------------------------------------------
    # 3. Build the ERASE mask
    #    = old garment area  +  arm pixels that the warped garment covers
    #    Never erase: face/hair, lower body
    # ------------------------------------------------------------------
    # Dilate old garment mask slightly to catch stray fringe pixels
    if args.garment_dilation > 0:
        k = args.garment_dilation | 1
        dil_kernel = np.ones((k, k), np.uint8)
        old_garment_dilated = cv2.dilate(old_garment_mask.astype(np.uint8), dil_kernel, iterations=1).astype(np.float32)
    else:
        old_garment_dilated = old_garment_mask

    # Arms are erased only where the new warped garment physically lands on them
    arm_under_garment = arm_mask * warped_mask

    erase_mask = np.clip(old_garment_dilated + arm_under_garment, 0.0, 1.0)

    # Hard-protect: never erase face/hair/lower body regardless of above
    protection = np.clip(preserve_mask + lower_mask, 0.0, 1.0)
    erase_mask = erase_mask * (1.0 - protection)

    # ------------------------------------------------------------------
    # 4. Build GARMENT paste mask with soft feathered edge
    # ------------------------------------------------------------------
    # Restrict cloth to person silhouette so it doesn't bleed into background
    paste_mask_hard = warped_mask * (person_alpha[:, :, 0] > 0.1).astype(np.float32)
    # Also don't paste over face/hair/lower
    paste_mask_hard = paste_mask_hard * (1.0 - protection)

    paste_mask_soft = soft_mask(paste_mask_hard, blur_radius=args.feather)
    paste_mask_soft = np.clip(paste_mask_soft, 0.0, 1.0)[:, :, None]  # (H,W,1)

    # ------------------------------------------------------------------
    # 5. Layer composition  (all operations in float32 [0,1])
    #
    #   canvas = person_rgb with conservative erase (cloth-supported)
    #          → paste warped_cloth using soft mask
    #          → restore uncovered erase-holes from original (avoid gray blocks)
    #          → copy-back hard-preserve zone from original
    # ------------------------------------------------------------------
    erase_mask_3d = erase_mask[:, :, None]   # (H,W,1)
    paste_alpha = paste_mask_soft[:, :, 0]

    # Step A: erase old garment / covered arms → neutral mid-grey.
    # Only keep hard erase where there is cloth support nearby.
    support_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    paste_support = cv2.dilate((paste_alpha > 0.08).astype(np.uint8), support_kernel, iterations=1).astype(np.float32)
    erase_mask_3d = erase_mask_3d * paste_support[:, :, None]

    neutral = np.full_like(person_rgb, 0.5)
    canvas  = person_rgb * (1.0 - erase_mask_3d) + neutral * erase_mask_3d

    # Step B: paste warped garment with feathered soft mask
    canvas = canvas * (1.0 - paste_mask_soft) + warped_cloth * paste_mask_soft

    # Step B.1: if any erased region is not actually covered by cloth, restore original.
    # This removes visible gray patches caused by warp under-coverage.
    uncovered_holes = (erase_mask_3d[:, :, 0] > 0.15) & (paste_alpha < 0.12)
    if np.any(uncovered_holes):
        hole_blend = soft_mask(uncovered_holes.astype(np.float32), blur_radius=9)[:, :, None]
        hole_blend = np.clip(hole_blend, 0.0, 1.0)
        canvas = canvas * (1.0 - hole_blend) + person_rgb * hole_blend

    # Step C: copy-back hard-preserve zone (face, hair, glasses, hat)
    #         feathered slightly so boundary doesn't look hard-cut
    preserve_soft = soft_mask(preserve_mask, blur_radius=args.preserve_feather)[:, :, None]
    canvas = canvas * (1.0 - preserve_soft) + person_rgb * preserve_soft

    # ------------------------------------------------------------------
    # 6. Re-apply person alpha so background stays transparent
    #    (restore_background.py in phase 5 will composite onto original bg)
    # ------------------------------------------------------------------
    canvas_uint8 = np.clip(canvas * 255.0, 0, 255).astype(np.uint8)
    alpha_uint8  = np.clip(person_alpha[:, :, 0] * 255.0, 0, 255).astype(np.uint8)

    result_rgba = np.dstack([canvas_uint8, alpha_uint8])  # (H,W,4)
    result_img  = Image.fromarray(result_rgba, mode="RGBA")

    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    result_img.save(args.output_path)
    print(f"[SUCCESS] Layered try-on result saved to: {args.output_path}")

    # ------------------------------------------------------------------
    # 8. Debug / diagnostic saves (same folder, prefixed with dbg_)
    # ------------------------------------------------------------------
    dbg_dir = output_dir if output_dir else "."
    Image.fromarray((erase_mask * 255).astype(np.uint8)).save(
        os.path.join(dbg_dir, "dbg_erase_mask.png"))
    Image.fromarray((paste_mask_soft[:, :, 0] * 255).astype(np.uint8)).save(
        os.path.join(dbg_dir, "dbg_paste_mask.png"))
    if has_parse:
        Image.fromarray((preserve_mask * 255).astype(np.uint8)).save(
            os.path.join(dbg_dir, "dbg_preserve_mask.png"))
        Image.fromarray((arm_under_garment * 255).astype(np.uint8)).save(
            os.path.join(dbg_dir, "dbg_arm_erased.png"))
    Image.fromarray((paste_support * 255).astype(np.uint8)).save(
        os.path.join(dbg_dir, "dbg_paste_support.png"))
    print(f"[DEBUG] Diagnostic masks saved to: {dbg_dir}")


if __name__ == "__main__":
    main()
