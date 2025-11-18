import os

import cv2
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d

from utils.llamas_utils import get_horizontal_values_for_four_lanes
from tabulate import tabulate
import json
from tqdm import tqdm


def write_output_culane_format(lanes, output_path, ori_img_h, ori_img_w, cut_height=0):
    """
    Write lane detection output in CULane format
    
    Args:
        lanes: List of lane arrays, each [N, 2] with (x, y) coordinates
        output_path: Path to save the output file
        ori_img_h: Original image height
        ori_img_w: Original image width
        cut_height: Height to cut from top (default 0)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        for lane in lanes:
            # Handle different input formats
            if isinstance(lane, dict):
                # Legacy Polar R-CNN format: {'points': [...], ...}
                points = np.array(lane["points"], dtype=np.float32)
            elif isinstance(lane, np.ndarray):
                # AFPL-Net format: direct numpy array [N, 2]
                points = lane.astype(np.float32)
            elif isinstance(lane, list):
                # List format
                points = np.array(lane, dtype=np.float32)
            else:
                continue
            
            # Ensure 2D shape
            if points.ndim == 1:
                points = points.reshape(-1, 2)
            
            # Skip invalid lanes
            if len(points) < 2:
                continue
            
            # Extract x, y coordinates
            xs = points[:, 0]
            ys = points[:, 1]
            
            # Filter out invalid points
            valid_mask = (xs >= 0) & (xs < ori_img_w) & (ys >= 0) & (ys < ori_img_h)
            if not np.any(valid_mask):
                continue
            
            xs = xs[valid_mask]
            ys = ys[valid_mask]
            
            # Sort by y coordinate
            sort_idx = np.argsort(ys)
            xs = xs[sort_idx]
            ys = ys[sort_idx]
            
            # Create interpolation function
            try:
                lane_curve = interp1d(ys, xs, fill_value="extrapolate")
            except ValueError:
                # Skip if interpolation fails (e.g., duplicate y values)
                continue
            
            # Sample at standard y positions (every 10 pixels from cut_height to bottom)
            sample_ys = np.arange(cut_height, ori_img_h, 10)
            
            # Only use y values within the lane's range
            min_y, max_y = ys.min(), ys.max()
            sample_ys = sample_ys[(sample_ys >= min_y) & (sample_ys <= max_y)]
            
            if len(sample_ys) == 0:
                continue
            
            # Interpolate x values
            sample_xs = lane_curve(sample_ys)
            
            # Write to file in CULane format: x1 y1 x2 y2 x3 y3 ...
            for x, y in zip(sample_xs, sample_ys):
                f.write(f"{x:.5f} {y:.5f} ")
            f.write('\n')


def draw_lane(lane, img=None, img_shape=None, width=30):
    if img is None:
        img = np.zeros(img_shape, dtype=np.uint8)
    lane = lane.astype(np.int32)
    for p1, p2 in zip(lane[:-1], lane[1:]):
        cv2.line(img, tuple(p1), tuple(p2), color=(255, 255, 255), thickness=width)
    return img

def discrete_cross_iou(xs, ys, width=30, ori_img_h=224, ori_img_w=224):
    xs = [draw_lane(lane, img_shape=(ori_img_h, ori_img_w, 3), width=width) > 0 for lane in xs]
    ys = [draw_lane(lane, img_shape=(ori_img_h, ori_img_w, 3), width=width) > 0 for lane in ys]
    ious = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            ious[i, j] = (x & y).sum() / (x | y).sum()
    return ious

def interp(points, n=50):
    x = [x for x, _ in points]
    y = [y for _, y in points]
    tck, u = splprep([x, y], s=0, t=n, k=min(3, len(points) - 1))

    u = np.linspace(0., 1., num=(len(u) - 1) * n + 1)
    return np.array(splev(u, tck)).T

def interp(points, n=50, is_equal_n=False):
    if n == 1:
        return np.array(points)
    x = [x for x, _ in points]
    y = [y for _, y in points]
    tck, u = splprep([x, y], s=0, t=n, k=min(3, len(points) - 1))
    if is_equal_n:
        u = np.linspace(0., 1., num=n)
    else:
        u = np.linspace(0., 1., num=(len(u) - 1) * n + 1)
    return np.array(splev(u, tck)).T

def culane_metric(pred,
                  anno,
                  width=30,
                  ori_img_h = 224,
                  ori_img_w = 224,
                  iou_thresholds=[0.5]):
    _metric = {}
    for thr in iou_thresholds:
        tp = 0
        fp = 0 if len(anno) != 0 else len(pred)
        fn = 0 if len(pred) != 0 else len(anno)
        _metric[thr] = [tp, fp, fn]

    interp_pred = np.array([interp(pred_lane, n=5) for pred_lane in pred],
                           dtype=object)  # (4, 50, 2)
    interp_anno = np.array([interp(anno_lane, n=5) for anno_lane in anno],
                           dtype=object)  # (4, 50, 2)

    ious = discrete_cross_iou(interp_pred, interp_anno, width=width, ori_img_h=ori_img_h, ori_img_w=ori_img_w)
    row_ind, col_ind = linear_sum_assignment(1 - ious)

    _metric = {}
    for thr in iou_thresholds:
        tp = int((ious[row_ind, col_ind] > thr).sum())
        fp = len(pred) - tp
        fn = len(anno) - tp
        _metric[thr] = [tp, fp, fn]
    return _metric


def llamas_metric(pred,
                  anno,
                  width=30,
                  ori_img_h = 224,
                  ori_img_w = 224,
                  iou_thresholds=[0.5]):
    _metric = {}
    for thr in iou_thresholds:
        tp = 0
        fp = 0 if len(anno) != 0 else len(pred)
        fn = 0 if len(pred) != 0 else len(anno)
        _metric[thr] = [tp, fp, fn]

    interp_pred = np.array([interp(pred_lane, n=50, is_equal_n=True) for pred_lane in pred],
                           dtype=object)  # (4, 50, 2)
    anno = np.array([np.array(anno_lane) for anno_lane in anno], dtype=object)

    ious = discrete_cross_iou(interp_pred, anno, width=width, ori_img_h=ori_img_h, ori_img_w=ori_img_w)
    row_ind, col_ind = linear_sum_assignment(1 - ious)

    _metric = {}
    for thr in iou_thresholds:
        tp = int((ious[row_ind, col_ind] > thr).sum())
        fp = len(pred) - tp
        fn = len(anno) - tp
        _metric[thr] = [tp, fp, fn]
    return _metric

def stat_results(results, iou_thresholds=[0.5]):
    out = {}
    for thr in iou_thresholds:
        tp = sum(m[thr][0] for m in results)
        fp = sum(m[thr][1] for m in results)
        fn = sum(m[thr][2] for m in results)
        precision = float(tp) / (tp + fp) if tp != 0 else 0
        recall = float(tp) / (tp + fn) if tp != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if tp !=0 else 0
        out[thr] = {'TP': tp, 'FP': fp, 'FN': fn, 'Precision': precision, 'Recall': recall, 'F1': f1}
    return out

def load_data_culane_format(data_dir, label_name_list):
    
    label_list = []
    filepaths = [os.path.join(data_dir, label_name) for label_name in label_name_list]
    for path in filepaths:
        with open(path, 'r') as data_file:
            labels = data_file.readlines()
        labels = [line.split() for line in labels]
        labels = [list(map(float, lane)) for lane in labels]
        labels = [[(lane[i], lane[i + 1]) for i in range(0, len(lane), 2)]
                    for lane in labels]
        labels = [lane for lane in labels if len(lane) >= 2]
        label_list.append(labels)
    return label_list


def load_data_dlrail_format(data_dir, label_name_list):
    label_list = []
    filepaths = [os.path.join(data_dir, label_name) for label_name in label_name_list]
    for path in filepaths:
        with open(path, 'r') as data_file:
            labels = data_file.readlines()
        labels = [line.split()[1:] for line in labels]
        labels = [list(map(float, lane)) for lane in labels]
        labels = [[(lane[i], lane[i + 1]) for i in range(0, len(lane), 2)]
                    for lane in labels]
        labels = [lane for lane in labels if len(lane) >= 2]
        label_list.append(labels)
    return label_list


def add_ys(xs):
    """For each x in xs, make a tuple with x and its corresponding y."""
    xs = np.array(xs[300:])
    valid = xs >= 0
    xs = xs[valid]
    assert len(xs) > 1
    ys = np.arange(300, 717)[valid]
    return list(zip(xs, ys))

def load_data_llamas_format(data_dir, label_name_list):
    """Loads the annotations and its paths
    Each annotation is converted to a list of points (x, y)
    """
    label_paths = [os.path.join(data_dir, 'labels/valid', label_name.replace('.lines.txt', '.json'))  for label_name in label_name_list]
    # label_paths = get_files_from_folder(label_dir, '.json')
    annos = [
        [
            add_ys(xs) for xs in
            get_horizontal_values_for_four_lanes(label_path)
            if (np.array(xs) >= 0).sum() > 1
        ]  # lanes annotated with a single point are ignored
        for label_path in tqdm(label_paths, desc=f'Reading the labels')
    ]
    return np.array(annos, dtype=object)


def read_culane_data(data_dir, file_list_path):
    """
    Read CULane dataset file list
    
    Args:
        data_dir: Root directory of CULane dataset
        file_list_path: Path to the file list (e.g., list/test_split/test.txt)
        
    Returns:
        List of image paths relative to data_dir
    """
    with open(file_list_path, 'r') as f:
        file_list = [line.strip() for line in f.readlines() if line.strip()]
    return file_list


def evaluate_culane_predictions(pred_dir, gt_dir, file_list, iou_threshold=0.5):
    """
    Evaluate CULane predictions against ground truth
    
    This is a simplified evaluation function. 
    For official evaluation, use the CULane official evaluation scripts.
    
    Args:
        pred_dir: Directory containing prediction txt files
        gt_dir: Directory containing ground truth txt files
        file_list: List of image file paths (relative)
        iou_threshold: IoU threshold for matching (default 0.5)
        
    Returns:
        Dictionary with evaluation metrics
    """
    # This function would implement the evaluation logic
    # For now, it's a placeholder that should call the official CULane eval code
    pass











