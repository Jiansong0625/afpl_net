"""
LLAMAS dataset specific utilities
"""

import numpy as np


def get_horizontal_values_for_four_lanes(lanes, y_samples):
    """
    Get horizontal (x) values for lanes at specified y-coordinates.
    Specifically formatted for LLAMAS dataset which expects 4 lanes.
    
    Args:
        lanes: List of lane arrays, each with shape [N, 2] (x, y)
        y_samples: Array of y-coordinates to sample at
        
    Returns:
        List of 4 lists, each containing x-coordinates at y_samples
        Returns [-2, -2, ...] for invalid/missing lanes
    """
    # LLAMAS expects exactly 4 lanes
    result = []
    
    for i in range(4):
        if i < len(lanes):
            lane = lanes[i]
            if lane is None or len(lane) == 0:
                # Invalid lane
                result.append([-2] * len(y_samples))
                continue
            
            # Interpolate x values at y_samples
            x_values = []
            for y_sample in y_samples:
                # Find closest y-coordinate in lane
                if len(lane) == 0:
                    x_values.append(-2)
                    continue
                
                y_coords = lane[:, 1]
                
                # Check if y_sample is within lane range
                if y_sample < y_coords.min() or y_sample > y_coords.max():
                    x_values.append(-2)
                    continue
                
                # Linear interpolation
                idx = np.searchsorted(y_coords, y_sample)
                if idx == 0:
                    x_values.append(float(lane[0, 0]))
                elif idx >= len(lane):
                    x_values.append(float(lane[-1, 0]))
                else:
                    # Interpolate between idx-1 and idx
                    y1, y2 = y_coords[idx-1], y_coords[idx]
                    x1, x2 = lane[idx-1, 0], lane[idx, 0]
                    
                    if y2 - y1 < 1e-6:
                        x = x1
                    else:
                        x = x1 + (x2 - x1) * (y_sample - y1) / (y2 - y1)
                    
                    x_values.append(float(x))
            
            result.append(x_values)
        else:
            # No lane at this index
            result.append([-2] * len(y_samples))
    
    return result
