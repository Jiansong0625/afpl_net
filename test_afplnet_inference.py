"""
Inference test script for AFPL-Net

This script is specifically designed for AFPL-Net (Anchor-Free Polar Lane Network)
and performs inference on test datasets, outputting results for evaluation.

Key features:
- Optimized format conversion with minimal precision loss
- Robust coordinate normalization detection
- Intelligent point filtering (not clipping)

Usage:
    python test_afplnet_inference.py --cfg Config/afplnet_culane_r18.py --weight_path path/to/weights.pth
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from tools.get_config import get_cfg
from Dataset.build import build_testset
from Models.build import build_model
from Eval.build import build_evaluator
import argparse


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='AFPL-Net Inference Test')
    parser.add_argument('--gpu_no', default=0, type=int, help='GPU device ID')
    parser.add_argument('--test_batch_size', default=32, type=int, help='Batch size for testing')
    parser.add_argument('--cfg', default='./Config/afplnet_culane_r18.py', type=str, help='Config file path')
    parser.add_argument('--result_path', default='./result', type=str, help='Path to save results')
    parser.add_argument('--weight_path', default='', type=str, help='Path to model weights')
    parser.add_argument('--view_path', default='./view', type=str, help='Path to save visualizations')
    parser.add_argument('--is_view', default=0, type=int, help='Whether to save visualizations (0 or 1)')
    parser.add_argument('--is_val', default=0, type=int, help='Whether to use validation set (0 or 1)')
    args = parser.parse_args()
    return args


def main():
    # Parse arguments and load config
    cfg = get_cfg(parse_args())
    
    # Device selection: GPU if available and valid, otherwise CPU
    if torch.cuda.is_available() and 0 <= cfg.gpu_no < torch.cuda.device_count():
        device = torch.device(f"cuda:{cfg.gpu_no}")
        print(f"Using GPU: cuda:{cfg.gpu_no}")
    else:
        device = torch.device("cpu")
        print(f"[Warning] Using CPU inference. CUDA available={torch.cuda.is_available()}, "
              f"count={torch.cuda.device_count()}, gpu_no={cfg.gpu_no}")
    
    # Verify this is an AFPL-Net config
    if not hasattr(cfg, 'cfg_name') or 'afplnet' not in cfg.cfg_name.lower():
        print("[Warning] This script is designed for AFPL-Net configs.")
        print(f"Current config: {getattr(cfg, 'cfg_name', 'Unknown')}")
    
    # Build AFPL-Net model
    print("Building AFPL-Net model...")
    net = build_model(cfg)
    
    # Load weights
    if not cfg.weight_path or not os.path.isfile(cfg.weight_path):
        raise FileNotFoundError(f"Weight file not found: {cfg.weight_path}")
    
    print(f"Loading weights from: {cfg.weight_path}")
    state = torch.load(cfg.weight_path, map_location='cpu')
    net.load_state_dict(state, strict=True)
    net.to(device).eval()
    print("Model loaded successfully!")
    
    # Build test dataset
    print("Building test dataset...")
    tsset = build_testset(cfg)
    print(f'Test set length: {len(tsset)}')
    
    # Build evaluator
    print("Building evaluator...")
    evaluator = build_evaluator(cfg)
    evaluator.pre_process()
    
    # Create data loader
    tsloader = torch.utils.data.DataLoader(
        tsset, 
        batch_size=cfg.test_batch_size, 
        shuffle=False, 
        num_workers=0,  # Windows: keep 0 for stability
        drop_last=False, 
        collate_fn=tsset.collate_fn,
        pin_memory=(device.type == "cuda")
    )
    
    # Run inference
    mode = "visualization" if cfg.is_view else "evaluation"
    print(f"Starting inference on {len(tsset)} images in {mode} mode...")
    
    for i, (img, file_names, ori_imgs) in enumerate(tqdm(tsloader, desc='AFPL-Net inference')):
        with torch.inference_mode():
            img = img.to(device, non_blocking=True)
            # Forward pass through AFPL-Net
            pred_dict = net(img)
            
            # Format AFPL-Net output for evaluator
            outputs = format_afplnet_output(pred_dict, cfg)
        
        # Write or visualize output
        if cfg.is_view:
            evaluator.view_output(outputs, file_names, ori_imgs)
        else:
            evaluator.write_output(outputs, file_names)
    
    # Final evaluation or visualization
    if cfg.is_view:
        print("\nGenerating ground truth visualization...")
        evaluator.view_gt()
        print(f"Visualizations saved to: {cfg.view_path}")
    else:
        print("\nRunning evaluation...")
        evaluator.evaluate()
        print(f"Results saved to: {cfg.result_path}")
    
    print("\n✅ Inference completed successfully!")


def format_afplnet_output(pred_dict, cfg):
    """
    Format AFPL-Net output to match CULane evaluator expectations
    
    Optimized for accuracy:
    - Robust coordinate normalization detection
    - Point filtering instead of clipping (preserves lane shape)
    - Configurable minimum point threshold
    
    Args:
        pred_dict: Dictionary containing 'lane_list' key from AFPL-Net
        cfg: Configuration object
        
    Returns:
        dict: Dictionary with 'lane_list' key containing formatted lanes
    """
    # Extract lane_list from pred_dict
    lanes_batch = pred_dict.get('lane_list', [])
    
    # Get minimum points threshold from config (default 2 for CULane)
    min_points = getattr(cfg, 'min_lane_points', 2)
    
    formatted_batch = []
    for img_idx, lanes in enumerate(lanes_batch):
        formatted_lanes = []
        for lane_idx, lane in enumerate(lanes):
            # Extract points from various formats
            points = None
            
            if isinstance(lane, dict):
                # Dictionary format: {'points': [...], 'scores': [...], ...}
                if 'points' in lane:
                    points = np.array(lane['points'], dtype=np.float32)
                else:
                    continue
            elif isinstance(lane, np.ndarray):
                # Already numpy array
                points = lane.astype(np.float32)
            elif isinstance(lane, list):
                # List format
                points = np.array(lane, dtype=np.float32)
            else:
                continue
            
            # Ensure 2D array shape [N, 2]
            if points.ndim == 1:
                if len(points) % 2 == 0:
                    points = points.reshape(-1, 2)
                else:
                    continue
            
            # Validate shape
            if points.shape[1] != 2:
                continue
            
            # Skip lanes with too few points
            if len(points) < min_points:
                continue
            
            # Robust coordinate normalization detection
            # Method 1: Check config flag (most reliable)
            if hasattr(cfg, 'output_normalized') and cfg.output_normalized:
                points[:, 0] *= cfg.img_w
                points[:, 1] *= cfg.img_h
            # Method 2: Heuristic - if max coordinate < 10, likely normalized
            elif points.max() < 10.0:
                points[:, 0] *= cfg.img_w
                points[:, 1] *= cfg.img_h
            # If coordinates already in pixel range, no conversion needed
            
            # Filter out-of-bound points (better than clipping for accuracy)
            valid_mask = (
                (points[:, 0] >= 0) & (points[:, 0] < cfg.img_w) &
                (points[:, 1] >= 0) & (points[:, 1] < cfg.img_h)
            )
            points = points[valid_mask]
            
            # Re-check point count after filtering
            if len(points) < min_points:
                continue
            
            # Map coordinates from resized image space to original image space
            # This is critical for evaluation which expects original image coordinates
            if hasattr(cfg, 'ori_img_w') and hasattr(cfg, 'ori_img_h') and hasattr(cfg, 'cut_height'):
                # x: scale from img_w to ori_img_w
                points[:, 0] = points[:, 0] * (cfg.ori_img_w / cfg.img_w)
                # y: scale from img_h to (ori_img_h - cut_height), then add cut_height offset
                points[:, 1] = points[:, 1] * ((cfg.ori_img_h - cfg.cut_height) / cfg.img_h) + cfg.cut_height
            
            # Sort by y-coordinate (top to bottom) - CRITICAL for CULane format
            sort_idx = np.argsort(points[:, 1])
            points = points[sort_idx]
            
            # Append as pure numpy array
            formatted_lanes.append(points)
        
        formatted_batch.append(formatted_lanes)
    
    # Return dictionary with 'lane_list' key
    return {'lane_list': formatted_batch}


if __name__ == '__main__':
    main()
