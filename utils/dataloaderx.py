"""
DataLoaderX: Optimized DataLoader with prefetch_generator for faster data loading
"""

import torch
from torch.utils.data import DataLoader


class DataLoaderX(DataLoader):
    """
    Extended DataLoader with optional prefetch generator for improved performance.
    Falls back to standard DataLoader if prefetch_generator is not available.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def __iter__(self):
        # Try to use prefetch_generator if available
        try:
            from prefetch_generator import BackgroundGenerator
            return BackgroundGenerator(super().__iter__())
        except ImportError:
            # Fall back to standard iterator
            return super().__iter__()
