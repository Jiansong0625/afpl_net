"""
Utility functions for lane detection tasks
"""

import numpy as np
import cv2


def clipline_out_of_image(line_coords, img_shape):
    """
    Clip lane line coordinates to be within image boundaries.
    
    Args:
        line_coords: numpy array of shape [N, 2] with (x, y) coordinates
        img_shape: tuple of (height, width)
        
    Returns:
        numpy array of clipped coordinates or None if all points are outside
    """
    if line_coords is None or len(line_coords) == 0:
        return None
    
    h, w = img_shape[:2]
    
    # Filter points within image boundaries
    valid_mask = (
        (line_coords[:, 0] >= 0) & (line_coords[:, 0] < w) &
        (line_coords[:, 1] >= 0) & (line_coords[:, 1] < h)
    )
    
    clipped = line_coords[valid_mask]
    
    if len(clipped) == 0:
        return None
    
    return clipped


def points_to_lineseg(points, num_segments=6):
    """
    Convert a set of lane points to line segments.
    
    Args:
        points: numpy array of shape [N, 2] with (x, y) coordinates
        num_segments: number of segments to divide the lane into
        
    Returns:
        List of line segments, each segment is ((x1, y1), (x2, y2))
    """
    if points is None or len(points) < 2:
        return []
    
    # Sort points by y-coordinate
    sorted_points = points[np.argsort(points[:, 1])]
    
    if len(sorted_points) < 2:
        return []
    
    # Divide into segments
    segments = []
    segment_size = max(1, len(sorted_points) // num_segments)
    
    for i in range(0, len(sorted_points) - 1, segment_size):
        p1 = sorted_points[i]
        p2 = sorted_points[min(i + segment_size, len(sorted_points) - 1)]
        segments.append((tuple(p1), tuple(p2)))
    
    return segments
