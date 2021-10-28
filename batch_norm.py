"""
Paper: Revisiting Batch Normalization
arXiv Link: https://arxiv.org/pdf/2110.13989.pdf
Authors: Jim Davis and Logan Frank*
Affiliation: Department of Computer Science & Engineering, Ohio State University
Corresponding Email: frank.580@osu.edu (First: Logan, Last: Frank)
Date: Oct 25, 2021

This research was supported by the U.S. Air Force Research Laboratory under Contract #GRT00054740 (Release #AFRL-2021-3711). 
We would also like to thank the DoD HPCMP for the use of their computational resources.
"""

# PyTorch imports
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

class ScaleBatchNorm2d(_BatchNorm):
    """Our proposed Batch Normalization layer that incorporates alternative initialization of the scale parameter.

    Attributes:
        (All of the standard attributes for nn.BatchNorm2d, e.g., running_mean, running_var, etc.)
        weight: The affine transformation scale/weight parameter (the main focus of this work).
        bias: The affine transformation shift/bias parameter.
    """
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, bn_weight=1.0):
        """Initializes ScaleBatchNorm2d with the standard Batch Normalization parameters (num_features, eps, momentum, affine) and reinitializes the weight attribute."""
        super().__init__(num_features, eps, momentum, affine)

        if affine:
            bn_weight = 1.0 if bn_weight <= 0 else bn_weight

            nn.init.constant_(self.weight, bn_weight)            
            nn.init.constant_(self.bias, 0)

    def _check_input_dim(self, x):
        """Checks the dimension of the input tensor for correctness."""
        if x.dim() != 4:
            raise ValueError(f'expected 4D input (got {x.dim()}D input)')

    def forward(self, x):
        """Computes the forward pass using the default forward function of the super class (i.e., the default forward pass for nn.BatchNorm2d)."""
        return super().forward(x)

    def __repr__(self):
        return f'ScaleBatchNorm2d: {self.weight.mean()}, {self.bias.mean()}'

    def __str__(self):
        return f'ScaleBatchNorm2d: {self.weight.mean()}, {self.bias.mean()}'