# Visualization Mode Fix Summary

## Problem

When running AFPL-Net inference with the `--is_view` flag to enable visualization mode, the program would crash with a `KeyError: 'anchor_embeddings'`.

### Root Cause

1. **In `Eval/culane_evaluator.py` (lines 65-77)**: The `view_output()` method assumed that the `outputs` dictionary would always contain an `'anchor_embeddings'` key, which was true for the old Polar R-CNN structure.

2. **In `test_afplnet_inference.py`**: The `format_afplnet_output()` function only returns `{'lane_list': formatted_batch}` because AFPL-Net is anchor-free and doesn't use anchor embeddings.

3. **Secondary issue**: `format_afplnet_output()` was returning lanes as raw numpy arrays, but the visualization method `view_single_img_lane()` expects lanes in dictionary format with a `'points'` key.

## Solution

### 1. Make anchor visualization optional (`Eval/culane_evaluator.py`)

**Before:**
```python
def view_output(self, outputs, file_names, ori_imgs):
    line_paras_batch = outputs['anchor_embeddings'].copy()  # KeyError here!
    line_paras_batch[..., 0] *= math.pi
    line_paras_batch[..., 1] *= self.img_w
    lanes_list = outputs['lane_list']
    for lanes, line_paras, file_name, ori_img in zip(lanes_list, line_paras_batch, file_names, ori_imgs):
        # ... visualization code
```

**After:**
```python
def view_output(self, outputs, file_names, ori_imgs):
    lanes_list = outputs['lane_list']
    
    # Check if anchor_embeddings exists (for Polar R-CNN compatibility)
    # AFPL-Net is anchor-free and doesn't have anchor_embeddings
    has_anchors = 'anchor_embeddings' in outputs
    if has_anchors:
        line_paras_batch = outputs['anchor_embeddings'].copy()
        line_paras_batch[..., 0] *= math.pi
        line_paras_batch[..., 1] *= self.img_w
    
    for i, (lanes, file_name, ori_img) in enumerate(zip(lanes_list, file_names, ori_imgs)):
        # ... save lane prediction (always)
        
        # Save anchor visualization only if anchors exist
        if has_anchors:
            # ... save anchor visualization
```

### 2. Return correct lane format (`test_afplnet_inference.py`)

**Before:**
```python
# Append as pure numpy array
formatted_lanes.append(points)
```

**After:**
```python
# Append in dict format for compatibility with visualizer
# write_output_culane_format handles both formats, but view_single_img_lane needs dict
formatted_lanes.append({'points': points})
```

## Impact

### ✅ Benefits:
1. **Fixes the KeyError**: AFPL-Net can now run with `--is_view` flag without crashing
2. **Backward Compatible**: Polar R-CNN outputs with `anchor_embeddings` still work and get both lane and anchor visualizations
3. **Minimal Changes**: Only two small, surgical changes to the codebase
4. **Clear Separation**: AFPL-Net gets lane visualizations only (appropriate for anchor-free architecture), Polar R-CNN gets both

### 🎯 Expected Behavior:

- **AFPL-Net** (anchor-free):
  - Generates `{image_name}_pred.jpg` (lane predictions only)
  - No `{image_name}_anchor.jpg` (since it has no anchors)

- **Polar R-CNN** (anchor-based):
  - Generates both `{image_name}_pred.jpg` (lane predictions)
  - And `{image_name}_anchor.jpg` (anchor visualizations)

## Testing

Created two test files to verify the fix:

1. **`test_visualization_fix_simple.py`**: Lightweight logic verification
   - ✅ Verifies AFPL-Net output doesn't cause KeyError
   - ✅ Verifies Polar R-CNN backward compatibility
   - ✅ Verifies correct lane format

2. **`test_visualization_fix.py`**: Comprehensive test with mock data
   - Tests the complete flow with mock images and lanes
   - Validates file generation behavior

## Usage

```bash
# Now works without error!
python test_afplnet_inference.py \
    --cfg Config/afplnet_culane_r18.py \
    --weight_path path/to/weights.pth \
    --is_view 1
```

## Files Changed

1. `Eval/culane_evaluator.py`: Made anchor visualization conditional
2. `test_afplnet_inference.py`: Fixed lane format to use dict with 'points' key
