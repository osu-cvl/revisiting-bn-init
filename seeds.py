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

# Standard Python imports
import os
import random
import hashlib

# PyTorch imports
import torch

# Other imports
import numpy as np

def make_complex(simple_seed, verbose=False):
    """Transforms a simple integer into a more complex integer using MD5.

    Args:
        simple_seed: A simple integer (e.g., [0, 15]).
        verbose: A flag that decides if the original seed, new complex seed, and binary value of the complex seed will be displayed to console.

    Returns:
        A sufficiently large integer value with a balanced mix of 0's and 1's (in binary).
    """

    # Hash the simple seed to make high complexity representation: Good Practice in (Pseudo) Random Number Generation for Bioinformatics Applications, by David Jones (http://www0.cs.ucl.ac.uk/staff/d.jones/GoodPracticeRNG.pdf)
    m = hashlib.md5(str(simple_seed).encode('UTF-8'))

    # Convert to hex string
    hex_md5 = m.hexdigest()

    # Seed must be in range [0, 2**32 - 1], take last 32 bits (8 values * 4 bits per value = 32 bits)
    least_significant_32bits = hex_md5[-8:]

    # Convert hex to base 10
    complex_seed = int(least_significant_32bits, base=16)

    # If we want verbosity, display the value
    if verbose:
        print(f'Original seed: { simple_seed }, Complex seed: { complex_seed }, Binary value: { bin(complex_seed)[2:].zfill(32) }')

    # simple_seed in range [0, 88265] yields unique complex_seed values
    return complex_seed

def make_deterministic(seed):
    """Sets the seed for numerous random number generators (RNGs) and sets other flags for reproducibility.

    Args:
        seed: An integer value to be used as the seed for several RNGs. This value can be a simple integer (e.g., [0, 15]) as it will be made more complex.
    """

    seed = make_complex(seed)
    
    # NumPy
    np.random.seed(seed)
    np.random.default_rng(seed=seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Python / OS
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set other flags
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False