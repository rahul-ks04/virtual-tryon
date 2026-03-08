import os
import argparse
import subprocess
import shutil
import sys

def run_cmd(cmd, cwd=None):
    quoted_cmd = ['"' + c + '"' if ' ' in c else c for c in cmd]
    print("Executing: " + " ".join(quoted_cmd))
    # Use shell=True on Windows to help resolve commands like 'conda' in the PATH
    is_windows = os.name == 'nt'
    result = subprocess.run(cmd, cwd=cwd, shell=is_windows)
    if result.returncode != 0:
        print(f"Error executing command: {' '.join(cmd)}")
        return False
    return True

def main(person_image, garment_image, garment_type, output_root, project_root, schp_python="python", dp_python="python"):
    # Split the python commands into lists
    schp_python_cmd = schp_python.split()
    dp_python_cmd = dp_python.split()
    gen_python_cmd = ["python"] # Default for general steps
    
    # Setup output paths
    rembg_dir = os.path.join(output_root, "rembg")
    schp_dir = os.path.join(output_root, "schp")
    densepose_dir = os.path.join(output_root, "densepose")
    garment_dir = os.path.join(output_root, "garment")
    target_mask_dir = os.path.join(output_root, "target_mask")
    
    os.makedirs(output_root, exist_ok=True)
    
    # 1. Remove background from person image
    print("\n--- Step 1: Background Removal (rembg) ---")
    if not run_cmd(gen_python_cmd + [os.path.join(project_root, "src/remove_background.py"), 
                    "--input", person_image, "--output_dir", rembg_dir]):
        return

    person_no_bg = os.path.join(rembg_dir, "person.png")
    
    # 2. Run SCHP for parsing
    # SCHP simple_extractor processes a directory
    schp_input_dir = os.path.join(output_root, "temp_schp_input")
    if os.path.exists(schp_input_dir):
        shutil.rmtree(schp_input_dir)
    os.makedirs(schp_input_dir, exist_ok=True)
    shutil.copy(person_no_bg, os.path.join(schp_input_dir, "person.png"))
    
    print("\n--- Step 2: Human Parsing (SCHP) ---")
    # We pass the schp_python path to handle the specific conda environment
    if not run_cmd(schp_python_cmd + [os.path.join(project_root, "src/run_schp.py"), 
                    "--input_dir", schp_input_dir, "--output_dir", schp_dir, "--project_root", project_root]):
        return
    
    schp_mask = os.path.join(schp_dir, "person.png")
    
    # 3. Run DensePose
    print("\n--- Step 3: DensePose Extraction ---")
    if not run_cmd(dp_python_cmd + [os.path.join(project_root, "src/run_densepose.py"), 
                    "--input", person_no_bg, "--output_dir", densepose_dir, "--project_root", project_root]):
        return
    
    densepose_mask = os.path.join(densepose_dir, "person_densepose.png")
    
    # 4. Preprocess Garment
    print("\n--- Step 4: Garment Preprocessing ---")
    garment_cmd = gen_python_cmd + [os.path.join(project_root, "src/preprocess_garment.py"),
                   "--type", garment_type, "--input", garment_image, "--output_dir", garment_dir]
    if garment_type == "worn":
        garment_cmd += ["--schp_mask", schp_mask]
        
    if not run_cmd(garment_cmd):
        return
        
    # 5. Generate Target Mask
    print("\n--- Step 5: Target Mask Generation ---")
    if not run_cmd(gen_python_cmd + [os.path.join(project_root, "src/generate_target_mask.py"),
                    "--schp", schp_mask, "--densepose", densepose_mask, "--output_dir", target_mask_dir]):
        return

    # Clean up temp dir
    if os.path.exists(schp_input_dir):
        shutil.rmtree(schp_input_dir)

    print("\n" + "="*40)
    print("Preprocessing Pipeline Finished Successfully!")
    print(f"Find all results in: {output_root}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One-click Preprocessing Pipeline for Viton")
    parser.add_argument("--person", required=True, help="Path to user (person) image")
    parser.add_argument("--garment", required=True, help="Path to garment image")
    parser.add_argument("--type", choices=["flat", "worn"], default="flat", help="Garment image type")
    parser.add_argument("--output_root", default="d:/Final Project Viton/virtual-tryon/outputs")
    parser.add_argument("--project_root", default="d:/Final Project Viton/virtual-tryon")
    
    # Optional flags for environment-specific python paths
    parser.add_argument("--schp_py", default="python", help="Python command/path for SCHP env")
    parser.add_argument("--dp_py", default="python", help="Python command/path for DensePose env")
    
    args = parser.parse_args()
    main(args.person, args.garment, args.type, args.output_root, args.project_root, args.schp_py, args.dp_py)
