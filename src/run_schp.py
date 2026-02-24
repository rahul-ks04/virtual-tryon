import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import shutil

# This script is intended to be run in a consolidated environment
# It wraps the SCHP simple_extractor logic for easy pipeline use.

def setup_paths(project_root):
    """Adds necessary paths to sys.path for SCHP."""
    schp_path = os.path.join(project_root, 'Self-Correction-Human-Parsing')
    if schp_path not in sys.path:
        sys.path.append(schp_path)
    return schp_path

def run_inference(image_path, output_dir, project_root, model_path=None):
    """Runs SCHP inference on a single image."""
    schp_path = setup_paths(project_root)
    
    # Import SCHP modules (must be after path setup)
    try:
        from simple_extractor import main as schp_main
    except ImportError:
        print(f"Error: Could not import SCHP from {schp_path}. Check paths.")
        return

    # Default model path if not provided
    if not model_path:
        model_path = os.path.join(schp_path, 'checkpoints', 'exp-schp-201908301523-lip.pth')

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    # Prepare arguments for SCHP's simple_extractor
    # We mimic the argparse structure if necessary or call a modified main
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy image to a temp folder for SCHP processing
    temp_input = os.path.join(output_dir, 'temp_input')
    os.makedirs(temp_input, exist_ok=True)
    shutil.copy(image_path, temp_input)

    # SCHP simple_extractor usually takes --dataset, --model-restore, --input-dir, --output-dir
    # For simplicity, we directly call the main logic if possible or run via subprocess
    # Here we assume a slightly patched version or we use sys.argv trickery
    
    print(f"Running SCHP on {image_path}...")
    
    # We use subprocess to avoid polluted global states if SCHP isn't perfectly modular
    import subprocess
    cmd = [
        sys.executable, 
        os.path.join(schp_path, 'simple_extractor.py'),
        '--dataset', 'lip',
        '--model-restore', model_path,
        '--input-dir', temp_input,
        '--output-dir', output_dir
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("SCHP Inference failed:")
        print(result.stderr)
    else:
        print(f"SCHP Inference complete. Output in {output_dir}")
        
    # Cleanup temp
    shutil.rmtree(temp_input)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run SCHP Human Parsing')
    parser.add_argument('--input', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, required=True, help='Path to output directory')
    parser.add_argument('--root', type=str, required=True, help='Project root directory')
    parser.add_argument('--model', type=str, help='Path to SCHP model checkpoint')
    
    args = parser.parse_args()
    run_inference(args.input, args.output, args.root, args.model)
