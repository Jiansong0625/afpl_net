"""
Coordinate transformation utilities for polar coordinate conversion
"""

import numpy as np
import torch
import torch.nn.functional as F


class CoordTrans:
    """
    Coordinate transformation between image coordinates and polar (cartesian) coordinates.
    Used for converting between pixel coordinates and pole-relative coordinates.
    """
    
    def __init__(self, cfg):
        self.img_h = cfg.img_h
        self.img_w = cfg.img_w
        self.center_h = cfg.center_h
        self.center_w = cfg.center_w
        
        # Polar coordinate parameters
        self.num_offsets = getattr(cfg, 'num_offsets', 72)
        self.offset_stride = getattr(cfg, 'offset_stride', 4.507)
        
    def img2cartesian(self, lane_coords):
        """
        Convert image coordinates to cartesian (pole-relative) coordinates.
        
        Args:
            lane_coords: numpy array of shape [N, 2] with (x, y) in image space
            
        Returns:
            numpy array of shape [N, 2] with cartesian coordinates relative to pole
        """
        if lane_coords is None or len(lane_coords) == 0:
            return lane_coords
        
        # Create a copy to avoid modifying input
        cartesian = lane_coords.copy()
        
        # Translate origin to pole position
        cartesian[:, 0] = lane_coords[:, 0] - self.center_w
        cartesian[:, 1] = self.center_h - lane_coords[:, 1]  # Flip y-axis
        
        return cartesian
    
    def cartesian2img(self, cartesian_coords):
        """
        Convert cartesian (pole-relative) coordinates to image coordinates.
        
        Args:
            cartesian_coords: numpy array of shape [N, 2] with cartesian coordinates
            
        Returns:
            numpy array of shape [N, 2] with (x, y) in image space
        """
        if cartesian_coords is None or len(cartesian_coords) == 0:
            return cartesian_coords
        
        # Create a copy to avoid modifying input
        img_coords = cartesian_coords.copy()
        
        # Translate back to image coordinates
        img_coords[:, 0] = cartesian_coords[:, 0] + self.center_w
        img_coords[:, 1] = self.center_h - cartesian_coords[:, 1]  # Flip y-axis
        
        return img_coords
    
    def cartesian2polar(self, cartesian_coords):
        """
        Convert cartesian coordinates to polar coordinates (θ, r).
        
        Args:
            cartesian_coords: numpy array of shape [N, 2] with (x, y) in cartesian
            
        Returns:
            numpy array of shape [N, 2] with (theta, r) in polar
        """
        if cartesian_coords is None or len(cartesian_coords) == 0:
            return cartesian_coords
        
        x = cartesian_coords[:, 0]
        y = cartesian_coords[:, 1]
        
        theta = np.arctan2(y, x)
        r = np.sqrt(x**2 + y**2)
        
        return np.stack([theta, r], axis=1)
    
    def polar2cartesian(self, polar_coords):
        """
        Convert polar coordinates (θ, r) to cartesian coordinates.
        
        Args:
            polar_coords: numpy array of shape [N, 2] with (theta, r) in polar
            
        Returns:
            numpy array of shape [N, 2] with (x, y) in cartesian
        """
        if polar_coords is None or len(polar_coords) == 0:
            return polar_coords
        
        theta = polar_coords[:, 0]
        r = polar_coords[:, 1]
        
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        return np.stack([x, y], axis=1)


class CoordTrans_torch:
    """
    PyTorch version of coordinate transformation for use in neural networks.
    """
    
    def __init__(self, cfg):
        self.img_h = cfg.img_h
        self.img_w = cfg.img_w
        self.center_h = cfg.center_h
        self.center_w = cfg.center_w
        
        # Polar coordinate parameters
        self.num_offsets = getattr(cfg, 'num_offsets', 72)
        self.offset_stride = getattr(cfg, 'offset_stride', 4.507)
    
    def img2cartesian(self, lane_coords):
        """
        Convert image coordinates to cartesian (pole-relative) coordinates.
        
        Args:
            lane_coords: torch.Tensor of shape [N, 2] or [B, N, 2]
            
        Returns:
            torch.Tensor with cartesian coordinates
        """
        cartesian = lane_coords.clone()
        
        cartesian[..., 0] = lane_coords[..., 0] - self.center_w
        cartesian[..., 1] = self.center_h - lane_coords[..., 1]  # Flip y-axis
        
        return cartesian
    
    def cartesian2img(self, cartesian_coords):
        """
        Convert cartesian (pole-relative) coordinates to image coordinates.
        
        Args:
            cartesian_coords: torch.Tensor of shape [N, 2] or [B, N, 2]
            
        Returns:
            torch.Tensor with image coordinates
        """
        img_coords = cartesian_coords.clone()
        
        img_coords[..., 0] = cartesian_coords[..., 0] + self.center_w
        img_coords[..., 1] = self.center_h - cartesian_coords[..., 1]  # Flip y-axis
        
        return img_coords
    
    def cartesian2polar(self, cartesian_coords):
        """
        Convert cartesian coordinates to polar coordinates (θ, r).
        
        Args:
            cartesian_coords: torch.Tensor of shape [..., 2]
            
        Returns:
            torch.Tensor of shape [..., 2] with (theta, r)
        """
        x = cartesian_coords[..., 0]
        y = cartesian_coords[..., 1]
        
        theta = torch.atan2(y, x)
        r = torch.sqrt(x**2 + y**2)
        
        return torch.stack([theta, r], dim=-1)
    
    def polar2cartesian(self, polar_coords):
        """
        Convert polar coordinates (θ, r) to cartesian coordinates.
        
        Args:
            polar_coords: torch.Tensor of shape [..., 2] with (theta, r)
            
        Returns:
            torch.Tensor of shape [..., 2] with (x, y)
        """
        theta = polar_coords[..., 0]
        r = polar_coords[..., 1]
        
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        
        return torch.stack([x, y], dim=-1)
