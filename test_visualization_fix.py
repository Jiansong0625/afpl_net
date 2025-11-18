"""
Test script to verify visualization mode fix for AFPL-Net

This tests that the visualization mode works without 'anchor_embeddings' key
and that format_afplnet_output returns the correct format.
"""

import torch
import numpy as np
import sys
import os
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_afplnet_inference import format_afplnet_output
from Eval.culane_evaluator import CULaneEvaluator


class SimpleConfig:
    """Minimal config for testing"""
    def __init__(self):
        self.img_w = 1640
        self.img_h = 590
        self.ori_img_w = 1640
        self.ori_img_h = 590
        self.cut_height = 0
        self.min_lane_points = 2
        self.output_normalized = False


def test_format_afplnet_output():
    """Test that format_afplnet_output returns correct format"""
    print("Testing format_afplnet_output...")
    
    cfg = SimpleConfig()
    
    # Create mock prediction data - list of lanes for one image
    mock_lanes = [
        np.array([[100, 200], [120, 300], [140, 400]], dtype=np.float32),
        np.array([[200, 200], [220, 300], [240, 400]], dtype=np.float32),
    ]
    
    pred_dict = {
        'lane_list': [mock_lanes]  # Batch of 1 image
    }
    
    # Format output
    formatted = format_afplnet_output(pred_dict, cfg)
    
    # Check structure
    assert 'lane_list' in formatted, "Output must have 'lane_list' key"
    assert len(formatted['lane_list']) == 1, "Should have 1 batch item"
    assert len(formatted['lane_list'][0]) == 2, "Should have 2 lanes"
    
    # Check that lanes are in dict format with 'points' key
    for lane in formatted['lane_list'][0]:
        assert isinstance(lane, dict), f"Lane should be dict, got {type(lane)}"
        assert 'points' in lane, "Lane dict must have 'points' key"
        assert isinstance(lane['points'], np.ndarray), "Points should be numpy array"
        assert lane['points'].shape[1] == 2, "Points should be [N, 2] array"
    
    print("✓ format_afplnet_output test passed")
    return True


def test_view_output_without_anchors():
    """Test that view_output works without anchor_embeddings"""
    print("Testing view_output without anchor_embeddings...")
    
    # Create temporary directory for output
    temp_dir = tempfile.mkdtemp()
    
    try:
        class TestConfig:
            img_w = 1640
            img_h = 590
            ori_img_w = 1640
            ori_img_h = 590
            cut_height = 0
            data_root = temp_dir
            result_path = os.path.join(temp_dir, 'results')
            view_path = os.path.join(temp_dir, 'views')
            is_val = True
            is_view = True
        
        cfg = TestConfig()
        
        # Create evaluator
        evaluator = CULaneEvaluator(cfg)
        
        # Mock data - lanes in dict format (as returned by fixed format_afplnet_output)
        mock_lanes = [
            {'points': np.array([[100, 200], [120, 300], [140, 400]], dtype=np.float32)},
            {'points': np.array([[200, 200], [220, 300], [240, 400]], dtype=np.float32)},
        ]
        
        outputs = {
            'lane_list': [mock_lanes]  # Batch of 1 image
        }
        
        file_names = ['test_image.jpg']
        
        # Create mock RGB image
        ori_imgs = [torch.from_numpy(np.random.randint(0, 255, (590, 1640, 3), dtype=np.uint8))]
        
        # This should NOT crash even without anchor_embeddings
        try:
            evaluator.view_output(outputs, file_names, ori_imgs)
            print("✓ view_output test passed (without anchor_embeddings)")
            success = True
        except KeyError as e:
            if 'anchor_embeddings' in str(e):
                print(f"✗ view_output test FAILED: KeyError for anchor_embeddings")
                success = False
            else:
                raise
        
        return success
        
    finally:
        # Clean up
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def test_view_output_with_anchors():
    """Test that view_output still works WITH anchor_embeddings (Polar R-CNN compatibility)"""
    print("Testing view_output with anchor_embeddings (Polar R-CNN compatibility)...")
    
    # Create temporary directory for output
    temp_dir = tempfile.mkdtemp()
    
    try:
        class TestConfig:
            img_w = 1640
            img_h = 590
            ori_img_w = 1640
            ori_img_h = 590
            cut_height = 0
            data_root = temp_dir
            result_path = os.path.join(temp_dir, 'results')
            view_path = os.path.join(temp_dir, 'views')
            is_val = True
            is_view = True
        
        cfg = TestConfig()
        
        # Create evaluator
        evaluator = CULaneEvaluator(cfg)
        
        # Mock data with anchor_embeddings (Polar R-CNN format)
        mock_lanes = [
            {'points': np.array([[100, 200], [120, 300], [140, 400]], dtype=np.float32)},
            {'points': np.array([[200, 200], [220, 300], [240, 400]], dtype=np.float32)},
        ]
        
        # Mock anchor embeddings (theta, rho format)
        mock_anchors = np.array([
            [[0.5, 800], [0.6, 900]],  # 2 anchors for image 1
        ], dtype=np.float32)
        
        outputs = {
            'lane_list': [mock_lanes],
            'anchor_embeddings': mock_anchors
        }
        
        file_names = ['test_image.jpg']
        
        # Create mock RGB image
        ori_imgs = [torch.from_numpy(np.random.randint(0, 255, (590, 1640, 3), dtype=np.uint8))]
        
        # This should work and create both pred and anchor visualizations
        try:
            evaluator.view_output(outputs, file_names, ori_imgs)
            
            # Check that both files would be created
            pred_path = os.path.join(temp_dir, 'views', 'test_image_pred.jpg')
            anchor_path = os.path.join(temp_dir, 'views', 'test_image_anchor.jpg')
            
            # Files should exist
            if os.path.exists(pred_path) and os.path.exists(anchor_path):
                print("✓ view_output test passed (with anchor_embeddings)")
                success = True
            else:
                print(f"✗ Files not created. Pred: {os.path.exists(pred_path)}, Anchor: {os.path.exists(anchor_path)}")
                success = False
        except Exception as e:
            print(f"✗ view_output test FAILED with exception: {e}")
            success = False
        
        return success
        
    finally:
        # Clean up
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def main():
    """Run all tests"""
    print("=" * 60)
    print("Running Visualization Fix Tests")
    print("=" * 60)
    
    results = []
    
    # Test 1: format_afplnet_output returns correct structure
    results.append(("format_afplnet_output", test_format_afplnet_output()))
    
    # Test 2: view_output works without anchor_embeddings (AFPL-Net)
    results.append(("view_output without anchors", test_view_output_without_anchors()))
    
    # Test 3: view_output still works with anchor_embeddings (Polar R-CNN)
    results.append(("view_output with anchors", test_view_output_with_anchors()))
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1


if __name__ == '__main__':
    exit(main())
