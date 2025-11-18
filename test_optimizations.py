"""
Test script to validate AFPL-Net optimizations

This script tests:
1. All optimization modules load correctly
2. Forward pass works with different configurations
3. Loss computation works
4. Performance benchmarks
"""

import torch
import torch.nn as nn
import sys
import time
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def test_attention_modules():
    """Test attention mechanism modules"""
    print("\n" + "="*60)
    print("Testing Attention Modules")
    print("="*60)
    
    try:
        from Models.Neck.attention import ChannelAttention, SpatialAttention, CBAM, CoordAttention
        
        # Test shapes
        batch_size = 2
        channels = 64
        height, width = 40, 100
        x = torch.randn(batch_size, channels, height, width)
        
        # Channel Attention
        ca = ChannelAttention(channels, reduction=16)
        out_ca = ca(x)
        assert out_ca.shape == x.shape, f"Channel Attention shape mismatch: {out_ca.shape} vs {x.shape}"
        print("✅ Channel Attention: PASSED")
        
        # Spatial Attention
        sa = SpatialAttention(kernel_size=7)
        out_sa = sa(x)
        assert out_sa.shape == x.shape, f"Spatial Attention shape mismatch: {out_sa.shape} vs {x.shape}"
        print("✅ Spatial Attention: PASSED")
        
        # CBAM
        cbam = CBAM(channels, reduction=16)
        out_cbam = cbam(x)
        assert out_cbam.shape == x.shape, f"CBAM shape mismatch: {out_cbam.shape} vs {x.shape}"
        print("✅ CBAM: PASSED")
        
        # Coordinate Attention
        coord_att = CoordAttention(channels, reduction=32)
        out_coord = coord_att(x)
        assert out_coord.shape == x.shape, f"Coord Attention shape mismatch: {out_coord.shape} vs {x.shape}"
        print("✅ Coordinate Attention: PASSED")
        
        print("\n✅ All attention modules working correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ Attention modules test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_enhanced_fpn():
    """Test enhanced FPN with attention"""
    print("\n" + "="*60)
    print("Testing Enhanced FPN")
    print("="*60)
    
    try:
        from Models.Neck.fpn import FPN
        
        # Create mock config
        class Config:
            fpn_in_channel = [128, 256, 512]
            neck_dim = 64
            fpn_use_attention = True
        
        cfg = Config()
        fpn = FPN(cfg)
        
        # Test input
        batch_size = 2
        inputs = [
            torch.randn(batch_size, 128, 40, 100),  # P3
            torch.randn(batch_size, 256, 20, 50),   # P4
            torch.randn(batch_size, 512, 10, 25),   # P5
        ]
        
        outputs = fpn(inputs)
        
        assert len(outputs) == 3, f"FPN output count mismatch: {len(outputs)} vs 3"
        assert outputs[0].shape[1] == 64, f"FPN output channel mismatch: {outputs[0].shape[1]} vs 64"
        print(f"✅ FPN output shapes: P3={outputs[0].shape}, P4={outputs[1].shape}, P5={outputs[2].shape}")
        
        # Test without attention
        cfg.fpn_use_attention = False
        fpn_no_att = FPN(cfg)
        outputs_no_att = fpn_no_att(inputs)
        assert len(outputs_no_att) == 3, "FPN without attention failed"
        print("✅ FPN without attention: PASSED")
        
        print("\n✅ Enhanced FPN working correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ Enhanced FPN test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiscale_head():
    """Test multi-scale AFPL head"""
    print("\n" + "="*60)
    print("Testing Multi-Scale AFPL Head")
    print("="*60)
    
    try:
        from Models.Head.afpl_head_multiscale import MultiScaleAFPLHead
        
        # Create mock config
        class Config:
            img_w, img_h = 800, 320
            neck_dim = 64
            center_w, center_h = 400, 25
            initial_pole = (400, 25)
            use_adaptive_pole = False
            use_multiscale_head = True
            use_depthwise_conv = False
            conf_thres = 0.1
            centerness_thres = 0.1
            angle_cluster_eps = 0.035
            min_cluster_points = 10
        
        cfg = Config()
        head = MultiScaleAFPLHead(cfg)
        
        # Test input (FPN features)
        batch_size = 2
        feats = [
            torch.randn(batch_size, 64, 40, 100),  # P3
            torch.randn(batch_size, 64, 20, 50),   # P4
            torch.randn(batch_size, 64, 10, 25),   # P5
        ]
        
        # Forward pass
        pred_dict = head(feats)
        
        assert 'cls_pred' in pred_dict, "Missing cls_pred"
        assert 'centerness_pred' in pred_dict, "Missing centerness_pred"
        assert 'theta_pred' in pred_dict, "Missing theta_pred"
        assert 'r_pred' in pred_dict, "Missing r_pred"
        assert 'pole_xy' in pred_dict, "Missing pole_xy"
        
        print(f"✅ Predictions shape: cls={pred_dict['cls_pred'].shape}, "
              f"centerness={pred_dict['centerness_pred'].shape}, "
              f"theta={pred_dict['theta_pred'].shape}, "
              f"r={pred_dict['r_pred'].shape}")
        
        # Test post-processing
        lanes = head.post_process(pred_dict, downsample_factor=8)
        assert len(lanes) == batch_size, f"Post-processing batch size mismatch: {len(lanes)} vs {batch_size}"
        print(f"✅ Post-processing: {len(lanes)} batch items processed")
        
        print("\n✅ Multi-scale AFPL head working correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ Multi-scale head test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_enhanced_loss():
    """Test enhanced loss functions"""
    print("\n" + "="*60)
    print("Testing Enhanced Loss Functions")
    print("="*60)
    
    try:
        from Loss.afpl_loss_enhanced import (
            AdaptiveFocalLoss, EnhancedPolarRegressionLoss, EnhancedAFPLLoss
        )
        
        batch_size = 2
        h, w = 40, 100
        
        # Test Adaptive Focal Loss
        focal = AdaptiveFocalLoss(alpha=0.25, gamma=2.0)
        pred = torch.randn(batch_size, 1, h, w)
        target = torch.randint(0, 2, (batch_size, h, w)).float()
        loss_focal = focal(pred, target)
        assert loss_focal.numel() == 1, "Focal loss should be scalar"
        print(f"✅ Adaptive Focal Loss: {loss_focal.item():.4f}")
        
        # Test Enhanced Polar Regression Loss
        polar_loss = EnhancedPolarRegressionLoss(beta=1.0)
        theta_pred = torch.randn(batch_size, 1, h, w)
        r_pred = torch.rand(batch_size, 1, h, w)
        theta_target = torch.randn(batch_size, h, w)
        r_target = torch.rand(batch_size, h, w) * 100
        mask = torch.randint(0, 2, (batch_size, h, w)).float()
        
        loss_polar = polar_loss(theta_pred, r_pred, theta_target, r_target, mask)
        assert loss_polar.numel() == 1, "Polar loss should be scalar"
        print(f"✅ Enhanced Polar Regression Loss: {loss_polar.item():.4f}")
        
        # Test Enhanced AFPL Loss
        class Config:
            cls_loss_weight = 1.0
            centerness_loss_weight = 1.5
            regression_loss_weight = 2.0
            cls_loss_alpha = 0.25
            cls_loss_gamma = 2.0
            regression_beta = 1.0
            use_adaptive_pole = False
            use_adaptive_focal_loss = True
            use_enhanced_regression_loss = True
            theta_loss_weight = 1.0
            r_loss_weight = 1.0
        
        cfg = Config()
        afpl_loss = EnhancedAFPLLoss(cfg)
        
        pred_dict = {
            'cls_pred': torch.randn(batch_size, 1, h, w),
            'centerness_pred': torch.randn(batch_size, 1, h, w),
            'theta_pred': torch.randn(batch_size, 1, h, w),
            'r_pred': torch.rand(batch_size, 1, h, w),
            'pole_xy': torch.tensor([[400.0, 25.0], [400.0, 25.0]])
        }
        
        target_dict = {
            'cls_gt': torch.randint(0, 2, (batch_size, h, w)).float(),
            'centerness_gt': torch.rand(batch_size, h, w),
            'theta_gt': torch.randn(batch_size, h, w),
            'r_gt': torch.rand(batch_size, h, w) * 100
        }
        
        total_loss, loss_dict = afpl_loss(pred_dict, target_dict)
        assert 'loss' in loss_dict, "Missing total loss"
        assert 'loss_cls' in loss_dict, "Missing cls loss"
        assert 'loss_centerness' in loss_dict, "Missing centerness loss"
        assert 'loss_reg' in loss_dict, "Missing regression loss"
        
        print(f"✅ Total Loss: {total_loss.item():.4f}")
        print(f"   - Classification: {loss_dict['loss_cls'].item():.4f}")
        print(f"   - Centerness: {loss_dict['loss_centerness'].item():.4f}")
        print(f"   - Regression: {loss_dict['loss_reg'].item():.4f}")
        
        print("\n✅ Enhanced loss functions working correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ Enhanced loss test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_model():
    """Test full AFPL-Net with optimizations"""
    print("\n" + "="*60)
    print("Testing Full AFPL-Net Model")
    print("="*60)
    
    try:
        from Models.afpl_net import AFPLNet
        
        # Create mock config with optimizations enabled
        class Config:
            cfg_name = 'afplnet_test'
            img_w, img_h = 800, 320
            backbone = 'resnet18'
            pretrained = False
            neck = 'fpn'
            fpn_in_channel = [128, 256, 512]
            neck_dim = 64
            downsample_strides = [8, 16, 32]
            center_w, center_h = 400, 25
            initial_pole = (400, 25)
            use_adaptive_pole = True
            pole_head_dim = 64
            conf_thres = 0.1
            centerness_thres = 0.1
            angle_cluster_eps = 0.035
            min_cluster_points = 10
            
            # Optimizations
            fpn_use_attention = True
            use_multiscale_head = True
            use_depthwise_conv = True
        
        cfg = Config()
        model = AFPLNet(cfg)
        model.eval()
        
        # Test input
        batch_size = 2
        img = torch.randn(batch_size, 3, 320, 800)
        
        # Forward pass
        with torch.no_grad():
            pred_dict = model(img)
        
        assert 'lane_list' in pred_dict, "Missing lane_list in output"
        assert len(pred_dict['lane_list']) == batch_size, "Lane list batch size mismatch"
        
        print(f"✅ Model forward pass successful")
        print(f"✅ Detected lanes: {[len(lanes) for lanes in pred_dict['lane_list']]} per batch item")
        
        # Test with dict input (training mode)
        model.train()
        sample_batch = {'img': img}
        pred_dict_train = model(sample_batch)
        assert 'cls_pred' in pred_dict_train, "Missing predictions in training mode"
        print(f"✅ Training mode forward pass successful")
        
        print("\n✅ Full model working correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ Full model test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def benchmark_performance():
    """Benchmark inference performance"""
    print("\n" + "="*60)
    print("Performance Benchmark")
    print("="*60)
    
    try:
        from Models.afpl_net import AFPLNet
        
        class Config:
            cfg_name = 'afplnet_benchmark'
            img_w, img_h = 800, 320
            backbone = 'resnet18'
            pretrained = False
            neck = 'fpn'
            fpn_in_channel = [128, 256, 512]
            neck_dim = 64
            downsample_strides = [8, 16, 32]
            center_w, center_h = 400, 25
            initial_pole = (400, 25)
            use_adaptive_pole = True
            pole_head_dim = 64
            conf_thres = 0.1
            centerness_thres = 0.1
            angle_cluster_eps = 0.035
            min_cluster_points = 10
            fpn_use_attention = False
            use_multiscale_head = False
            use_depthwise_conv = False
        
        # Test baseline
        cfg_baseline = Config()
        model_baseline = AFPLNet(cfg_baseline)
        model_baseline.eval()
        
        # Test optimized
        cfg_opt = Config()
        cfg_opt.fpn_use_attention = True
        cfg_opt.use_multiscale_head = True
        cfg_opt.use_depthwise_conv = True
        model_opt = AFPLNet(cfg_opt)
        model_opt.eval()
        
        # Benchmark
        img = torch.randn(1, 3, 320, 800)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model_baseline(img)
                _ = model_opt(img)
        
        # Baseline timing
        times_baseline = []
        for _ in range(50):
            start = time.time()
            with torch.no_grad():
                _ = model_baseline(img)
            times_baseline.append(time.time() - start)
        
        # Optimized timing
        times_opt = []
        for _ in range(50):
            start = time.time()
            with torch.no_grad():
                _ = model_opt(img)
            times_opt.append(time.time() - start)
        
        avg_baseline = np.mean(times_baseline) * 1000
        avg_opt = np.mean(times_opt) * 1000
        
        print(f"Baseline model: {avg_baseline:.2f} ms/image ({1000/avg_baseline:.1f} FPS)")
        print(f"Optimized model: {avg_opt:.2f} ms/image ({1000/avg_opt:.1f} FPS)")
        print(f"Speed difference: {(avg_opt/avg_baseline - 1)*100:+.1f}%")
        
        # Count parameters
        params_baseline = sum(p.numel() for p in model_baseline.parameters())
        params_opt = sum(p.numel() for p in model_opt.parameters())
        
        print(f"\nBaseline parameters: {params_baseline/1e6:.2f}M")
        print(f"Optimized parameters: {params_opt/1e6:.2f}M")
        print(f"Parameter difference: {(params_opt/params_baseline - 1)*100:+.1f}%")
        
        print("\n✅ Benchmark completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Benchmark FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("AFPL-Net Optimization Test Suite")
    print("="*60)
    
    results = {}
    
    # Run tests
    results['Attention Modules'] = test_attention_modules()
    results['Enhanced FPN'] = test_enhanced_fpn()
    results['Multi-Scale Head'] = test_multiscale_head()
    results['Enhanced Loss'] = test_enhanced_loss()
    results['Full Model'] = test_full_model()
    results['Performance Benchmark'] = benchmark_performance()
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
    
    print("="*60)
    print(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All optimizations working correctly! 🎉")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
