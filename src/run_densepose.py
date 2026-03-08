import os
import cv2
import torch
import numpy as np
import argparse
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from densepose import add_densepose_config
from densepose.vis.densepose_results import (
    DensePoseResultsFineSegmentationVisualizer,
)
from densepose.vis.extractor import DensePoseResultExtractor

def run_densepose(input_path, output_dir, project_root):
    # Setup paths
    config_file = os.path.join(project_root, "detectron2", "projects", "DensePose", "configs", "densepose_rcnn_R_50_FPN_s1x.yaml")
    model_weights = os.path.join(project_root, "detectron2", "checkpoints", "densepose_r50_fpn.pkl")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filename = os.path.basename(input_path)
    # Ensure output is png
    output_filename = os.path.splitext(filename)[0] + "_densepose.png"
    output_image_path = os.path.join(output_dir, output_filename)

    # Setup configuration
    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.freeze()

    # Load image
    print(f"Loading image: {input_path}")
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Error: Could not read image at {input_path}")

    # Create predictor
    print("Initializing DensePose model...")
    predictor = DefaultPredictor(cfg)

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        outputs = predictor(img)["instances"]

    # Visualize results
    print("Visualizing results...")
    visualizer = DensePoseResultsFineSegmentationVisualizer()
    extractor = DensePoseResultExtractor()
    
    # Process outputs
    results = extractor(outputs)
    
    # Create black background for visualization
    background = np.zeros_like(img)
    
    vis_img = visualizer.visualize(background, results)

    # Save output
    cv2.imwrite(output_image_path, vis_img)
    print(f"Result saved to: {output_image_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--project_root", required=True)
    args = parser.parse_args()
    run_densepose(args.input, args.output_dir, args.project_root)
