# AFPL-Net: Anchor-Free Polar Lane Detection Network

## Overview

AFPL-Net is a **single-stage, anchor-free** lane detection network that uses polar coordinates for efficient and accurate lane detection. It's designed for autonomous driving applications and tested on standard benchmarks like CULane.

### Key Features

- ✅ **Anchor-Free**: No need for predefined anchor patterns
- ✅ **Single-Stage**: Direct lane prediction without proposal generation
- ✅ **NMS-Free**: Uses angular clustering instead of Non-Maximum Suppression
- ✅ **Polar Coordinates**: Leverages vanishing point (pole) for geometric constraints
- ✅ **Centerness-Aware**: Predicts point quality for better lane localization

## Architecture

```
Input Image (3×320×800)
    ↓
Backbone (ResNet18/DLA34)
    ↓
FPN Neck (Multi-scale Features)
    ↓
AFPL Head (3 Parallel Branches)
    ├─ Classification Head → Lane/Non-lane
    ├─ Centerness Head → Point Quality
    └─ Polar Regression Head → (θ, r) Coordinates
    ↓
Post-Processing (Angular Clustering)
    ↓
Lane Detection Results
```

### Components

#### 1. **Backbone**
- Extracts multi-scale features from input images
- Supports: ResNet (18/34/50), DLA34, ConvNeXt, InceptionNeXt
- Pre-trained on ImageNet

#### 2. **FPN Neck**
- Feature Pyramid Network for multi-scale fusion
- Produces P3, P4, P5 feature maps
- Default output: 64 channels

#### 3. **AFPL Head** (3 Parallel Branches)

**a) Classification Branch**
- Predicts if each pixel belongs to a lane line
- Loss: Focal Loss (handles class imbalance)
- Output: [B, 1, H, W]

**b) Centerness Branch**
- Predicts quality of lane point (1.0 at center, 0.0 at edges)
- Loss: Binary Cross-Entropy
- Output: [B, 1, H, W]

**c) Polar Regression Branch**
- Predicts (θ, r) polar coordinates relative to vanishing point
- θ: angle ∈ [-π, π]
- r: distance ∈ [0, ∞)
- Loss: Periodic L1 (for θ) + Smooth L1 (for r)
- Output: [B, 2, H, W]

#### 4. **Post-Processing**
- **Angular Clustering**: Group points with similar θ using DBSCAN
- **NMS-Free**: No need for traditional Non-Maximum Suppression
- Configurable parameters:
  - `angle_cluster_eps`: Clustering tolerance (~2 degrees)
  - `min_cluster_points`: Minimum points per lane

## Installation

### Requirements

```bash
# Python 3.7+
pip install torch torchvision  # PyTorch 1.8+
pip install opencv-python
pip install albumentations
pip install scipy
pip install scikit-learn  # For DBSCAN clustering
pip install tqdm
```

### Optional
```bash
pip install prefetch_generator  # For faster data loading
pip install tensorboard  # For training visualization
```

## Usage

### Training

```bash
python train.py --cfg Config/afplnet_culane_r18.py --save_path work_dir/ckpt
```

### Inference

```bash
python test_afplnet_inference.py \
    --cfg Config/afplnet_culane_r18.py \
    --weight_path work_dir/ckpt/para_31.pth \
    --result_path ./result
```

### Visualization

```bash
python test_afplnet_inference.py \
    --cfg Config/afplnet_culane_r18.py \
    --weight_path work_dir/ckpt/para_31.pth \
    --is_view 1 \
    --view_path ./view
```

## Configuration

Key parameters in `Config/afplnet_culane_r18.py`:

```python
# Network Architecture
backbone = 'resnet18'
neck = 'fpn'
neck_dim = 64

# Global Pole (Vanishing Point)
center_h = 25  # y-coordinate
center_w = 386  # x-coordinate
use_adaptive_pole = True  # Learn pole per image

# Training
batch_size = 16
epoch_num = 32
lr = 6e-4

# Loss Weights
cls_loss_weight = 1.0
centerness_loss_weight = 1.5
regression_loss_weight = 2.0

# Post-Processing
conf_thres = 0.1
centerness_thres = 0.1
angle_cluster_eps = 0.035  # ~2 degrees
min_cluster_points = 10
```

## Network Design Rationale

### Why Polar Coordinates?

Lane lines naturally converge toward a vanishing point (horizon). Polar coordinates (θ, r) relative to this point provide:
1. **Geometric Constraints**: Lanes with similar angles are grouped together
2. **Natural Clustering**: Angular similarity directly corresponds to lane identity
3. **Efficient Representation**: Two values (θ, r) encode lane direction and position

### Why Anchor-Free?

Traditional anchor-based methods require:
- Predefined anchor patterns
- Complex anchor matching
- Difficult hyperparameter tuning

AFPL-Net simplifies this:
- Every pixel predicts directly
- No anchor design needed
- More flexible for various lane shapes

### Why NMS-Free?

Traditional NMS has issues:
- Hard to tune threshold
- May suppress valid detections
- Not end-to-end differentiable

Angular clustering is better:
- Automatic grouping by θ similarity
- No manual threshold tuning
- More robust to crowded lanes

## Recent Fixes and Optimizations

### Critical Bug Fixes
1. **Fixed `get_lanes()` method**: Corrected return key from `'lanes'` to `'lane_list'`
2. **Improved input handling**: Now safely handles both dict and tensor inputs
3. **Fixed syntax warnings**: Used raw strings for Windows paths

### Optimizations
1. **Enhanced centerness computation**: Improved numerical stability
2. **Better post-processing**: Added safety checks for edge cases
3. **Clearer code comments**: Improved documentation throughout

### Added Missing Utils Module
Created complete `utils/` package:
- `dataloaderx.py`: Optimized data loading
- `coord_transform.py`: Coordinate transformations
- `lane_utils.py`: Lane processing utilities
- `ploter.py`: Visualization tools
- `llamas_utils.py`: LLAMAS dataset support

## Evaluation

AFPL-Net is evaluated on:
- **CULane**: Large-scale challenging dataset (88,880 training images)
- **TuSimple**: Highway lane detection
- **LLAMAS**: Multi-lane highway dataset
- **CurveLanes**: Curved lane detection

Metrics:
- F1 Score
- Precision & Recall
- False Positive Rate

## Network Feasibility for Lane Detection

### ✅ **FEASIBLE and RECOMMENDED**

**Strengths:**
1. **Efficient Architecture**: Single-stage design is fast and deployable
2. **Robust to Occlusion**: Centerness helps identify reliable points
3. **Handles Multiple Lanes**: Angular clustering naturally separates lanes
4. **Geometric Reasoning**: Polar coordinates encode domain knowledge
5. **Well-Designed Losses**: Focal + Centerness + Polar regression work well together

**Suitable For:**
- Autonomous driving (highway & urban)
- Lane keeping assistance
- Road marking detection
- Real-time applications (with lightweight backbone)

**Limitations:**
1. Requires visible vanishing point (struggles with top-down views)
2. Single-scale prediction (could use multi-scale for small lanes)
3. Fixed pole location (adaptive pole helps but adds complexity)

## Citation

If you use this code, please cite the original Polar R-CNN work and related methods:

```bibtex
@inproceedings{polarrcnn,
  title={Polar R-CNN: Accurate Lane Detection via Geometry-Based Approach},
  author={...},
  booktitle={...},
  year={...}
}
```

## License

This project is for research and educational purposes.

## Contributing

Contributions are welcome! Please:
1. Test your changes thoroughly
2. Add comments for complex code
3. Update documentation
4. Follow existing code style

## Contact

For questions or issues, please open a GitHub issue.
