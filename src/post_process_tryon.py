import os
import argparse
import subprocess
import sys

def run_cmd(cmd):
    print(f"Executing: {' '.join(cmd)}")
    is_windows = os.name == 'nt'
    result = subprocess.run(cmd, shell=is_windows)
    if result.returncode != 0:
        print(f"Error executing command: {' '.join(cmd)}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Post-Processing Pipeline for VITON: Agnostic Generation & Composition")
    parser.add_argument("--person_image", required=True, help="Path to person image (no-bg)")
    parser.add_argument("--schp_mask", required=True, help="Path to SCHP mask")
    parser.add_argument("--warped_garment", required=True, help="Path to warped garment from FVNT")
    parser.add_argument("--warped_mask", required=True, help="Path to warped garment mask")
    parser.add_argument("--output_root", required=True, help="Root directory for outputs")
    parser.add_argument("--garment_type", choices=["full_sleeve", "top", "short_sleeve"], default="full_sleeve", 
                        help="Type of target garment (determines arm masking strategy)")
    parser.add_argument("--project_root", default="d:/Final Project Viton/virtual-tryon")
    
    args = parser.parse_args()

    # Setup paths
    agnostic_dir = os.path.join(args.output_root, "agnostic")
    tryon_dir = os.path.join(args.output_root, "tryon")
    python_exe = sys.executable

    # Strategic mode selection
    # If it's a 'top' or 'short_sleeve', preserve the original skin details.
    preserve_arms_cmd = []
    if args.garment_type in ["top", "short_sleeve"]:
        preserve_arms_cmd = ["--preserve_arms"]
        print(f"Setting Mode: PRESERVE ARMS (Optimized for {args.garment_type})")
    else:
        print(f"Setting Mode: STANDARD AGNOSTIC (Optimized for {args.garment_type})")

    # 1. Generate Agnostic Person
    print("\n--- Step 1: Agnostic Person Generation ---")
    agnostic_gen_cmd = [python_exe, os.path.join(args.project_root, "src/generate_agnostic_person.py"),
                        "--image", args.person_image, "--parse", args.schp_mask, "--output_dir", agnostic_dir]
    if preserve_arms_cmd:
        agnostic_gen_cmd += preserve_arms_cmd

    if not run_cmd(agnostic_gen_cmd):
        return

    # 2. Final Composition
    print("\n--- Step 2: Final Composition ---")
    agnostic_img = os.path.join(agnostic_dir, os.path.basename(args.person_image))
    if not run_cmd([python_exe, os.path.join(args.project_root, "src/compose_tryon.py"),
                    "--agnostic", agnostic_img, "--warped_garment", args.warped_garment,
                    "--warped_mask", args.warped_mask, "--output_dir", tryon_dir]):
        return

    print("\n" + "="*40)
    print("Post-processing Finished Successfully!")
    print(f"Final Try-On image: {os.path.join(tryon_dir, os.path.basename(args.person_image))}")
    print("="*40)

if __name__ == "__main__":
    main()
