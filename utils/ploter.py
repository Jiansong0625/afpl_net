"""
Visualization utilities for lane detection
"""

import numpy as np
import cv2


# Standard colors for visualization (BGR format)
COLORS = [
    (0, 0, 255),    # Red
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
    (255, 255, 0),  # Cyan
    (128, 0, 128),  # Purple
    (255, 128, 0),  # Orange
]


class Ploter:
    """
    Utility class for visualizing lane detection results.
    """
    
    def __init__(self, cfg=None):
        self.cfg = cfg
        self.img_h = cfg.img_h if cfg else 320
        self.img_w = cfg.img_w if cfg else 800
        self.center_h = cfg.center_h if cfg and hasattr(cfg, 'center_h') else 25
        self.center_w = cfg.center_w if cfg and hasattr(cfg, 'center_w') else 386
        
    def plot_lanes(self, img, lanes, color=(0, 255, 0), thickness=2):
        """
        Plot lane points on image.
        
        Args:
            img: Input image (numpy array)
            lanes: List of lane coordinates, each as [N, 2] array
            color: BGR color tuple
            thickness: Line thickness
            
        Returns:
            Image with lanes drawn
        """
        img_out = img.copy()
        
        for lane in lanes:
            if lane is None or len(lane) < 2:
                continue
            
            # Draw lane as connected line segments
            pts = lane.astype(np.int32)
            for i in range(len(pts) - 1):
                cv2.line(img_out, tuple(pts[i]), tuple(pts[i+1]), color, thickness)
        
        return img_out
    
    def plot_lanes_xs(self, img, lane_xs, lane_masks, color=(0, 255, 0), thickness=2):
        """
        Plot lanes from x-coordinates at fixed y-positions.
        
        Args:
            img: Input image (numpy array)
            lane_xs: Array of shape [num_lanes, num_points] with x coordinates
            lane_masks: Array of shape [num_lanes, num_points] indicating valid points
            color: BGR color tuple
            thickness: Line thickness
            
        Returns:
            Image with lanes drawn
        """
        img_out = img.copy()
        
        # Sample y coordinates
        num_points = lane_xs.shape[1] if len(lane_xs.shape) > 1 else len(lane_xs)
        sample_y = np.linspace(self.center_h, self.img_h, num_points)
        
        for i, (xs, mask) in enumerate(zip(lane_xs, lane_masks)):
            valid_points = []
            for x, y, valid in zip(xs, sample_y, mask):
                if valid > 0.5:
                    valid_points.append([int(x), int(y)])
            
            if len(valid_points) < 2:
                continue
            
            # Draw lane
            pts = np.array(valid_points, dtype=np.int32)
            for j in range(len(pts) - 1):
                cv2.line(img_out, tuple(pts[j]), tuple(pts[j+1]), 
                        COLORS[i % len(COLORS)] if color == (0, 255, 0) else color, 
                        thickness)
        
        return img_out
    
    def plot_lines(self, img, line_paras, color=(255, 0, 0), thickness=2):
        """
        Plot lines from line parameters (polar or parametric form).
        
        Args:
            img: Input image (numpy array)
            line_paras: Array of shape [N, 2] with line parameters
            color: BGR color tuple
            thickness: Line thickness
            
        Returns:
            Image with lines drawn
        """
        img_out = img.copy()
        
        for i, params in enumerate(line_paras):
            # Assuming polar form: (theta, rho)
            theta, rho = params
            
            # Convert polar to cartesian line equation
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho + self.center_w
            y0 = self.center_h - b * rho
            
            # Calculate line endpoints
            scale = max(self.img_h, self.img_w)
            x1 = int(x0 + scale * (-b))
            y1 = int(y0 + scale * (a))
            x2 = int(x0 - scale * (-b))
            y2 = int(y0 - scale * (a))
            
            # Draw line
            cv2.line(img_out, (x1, y1), (x2, y2), 
                    COLORS[i % len(COLORS)] if color == (255, 0, 0) else color, 
                    thickness)
        
        return img_out
    
    def plot_lines_group(self, img, line_paras_group, num_group=6, thickness=2):
        """
        Plot grouped line segments.
        
        Args:
            img: Input image (numpy array)
            line_paras_group: Array of shape [num_lanes, num_groups, 2]
            num_group: Number of groups
            thickness: Line thickness
            
        Returns:
            Image with lines drawn
        """
        img_out = img.copy()
        
        for lane_idx, groups in enumerate(line_paras_group):
            color = COLORS[lane_idx % len(COLORS)]
            
            for group_params in groups:
                if np.all(group_params == 0):
                    continue
                
                # Plot each group segment
                theta, rho = group_params
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho + self.center_w
                y0 = self.center_h - b * rho
                
                scale = self.img_h // num_group
                x1 = int(x0 + scale * (-b))
                y1 = int(y0 + scale * (a))
                x2 = int(x0 - scale * (-b))
                y2 = int(y0 - scale * (a))
                
                cv2.line(img_out, (x1, y1), (x2, y2), color, thickness)
        
        return img_out
    
    def plot_pole(self, img, color=(255, 255, 0), radius=5):
        """
        Draw the vanishing point (pole) on the image.
        
        Args:
            img: Input image (numpy array)
            color: BGR color tuple
            radius: Circle radius
            
        Returns:
            Image with pole marked
        """
        img_out = img.copy()
        cv2.circle(img_out, (int(self.center_w), int(self.center_h)), 
                  radius, color, -1)
        return img_out
