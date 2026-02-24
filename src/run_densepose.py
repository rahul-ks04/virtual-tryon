import os
import sys
import argparse
import subprocess
import torch
from PIL import Image

# This script wraps the Detectron2/DensePose inference logic.

def run_inference(image_path, output_dir, project_root):
    """Runs DensePose inference on a single image."""
    # We assume 'detectron2' is installed and 'densepose' is in the project root
    # or accessible.
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Path to the apply_net script in the detectron2/projects/DensePose folder
    # In the user's project, the detectron2 repo might be cloned inside virtual_tryon_project
    # Based on previous viewed files, we look for it.
    
    detectron2_path = os.path.join(project_root, 'detectron2')
    apply_net_path = os.path.join(detectron2_path, 'projects', 'DensePose', 'apply_net.py')
    
    if not os.path.exists(apply_net_path):
        # Alternative location check
        apply_net_path = os.path.join(project_root, 'projects', 'DensePose', 'apply_net.py')
        if not os.path.exists(apply_net_path):
            print(f"Error: DensePose apply_net.py not found. Expected at {apply_net_path}")
            return

    # Configuration and Model Checkpoint
    # These should ideally be in a standard location
    config_path = os.path.join(detectron2_path, 'projects', 'DensePose', 'configs', 'densepose_rcnn_R_50_FPN_s1x.yaml')
    model_checkpoint = os.path.join(project_root, 'model', 'densepose_rcnn_R_50_FPN_s1x.pkl')
    
    if not os.path.exists(model_checkpoint):
        print(f"Warning: DensePose checkpoint not found at {model_checkpoint}. Inference might fail.")

    print(f"Running DensePose on {image_path}...")
    
    # Output file names
    output_image = os.path.join(output_dir, 'densepose_res.png')
    
    cmd = [
        sys.executable,
        apply_net_path,
        'show',
        config_path,
        model_checkpoint,
        image_path,
        'dp_segm',
        '-v',
        '--output', output_image
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("DensePose Inference failed:")
        print(result.stderr)
    else:
        print(f"DensePose Inference complete. Result saved to {output_image}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run DensePose Surface Mapping')
    parser.add_argument('--input', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, required=True, help='Path to output directory')
    parser.add_argument('--root', type=str, required=True, help='Project root directory')
    
    args = parser.parse_args()
    run_inference(args.input, args.output, args.root)
