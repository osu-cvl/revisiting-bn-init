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

# T-Test import
from scipy.stats import ttest_rel

def evaluate(d1, d2):
    """Computes a one-sided paired t-test between two distributions.

    This function will take two lists / arrays (that represent individual instances in a distribution) and compute a statistical one-sided paired t-test. More specifically,
    it compares d1 to d2 to see if d1 is statistically significantly greater than d2. This function is not called in this code base, but since it is used in for the results
    of our paper, we include it for completeness.

    Args:
        d1: A collection of individual measured values (list, ndarray, tensor, etc.) that form a distribution. This parameter should represent our proposed approach.
        d2: A collection of individual measured values (list, ndarray, tensor, etc.) that form a distribution. This parameter should represent the baseline or other approach being compared to.

    Returns:
        A float p-value associated with a t-test. In our work, a value less than or equal to 0.05 signifies d1 is statistically significantly greater than d2 (i.e., d1 is a statistically significant improvement over d2)

    Raises:
        Exception: The lengths of d1 and d2 must be the same because this function computes a *paired* t-test
    """

    assert len(d1) == len(d2), 'Arrays d1 and d2 must be same length'

    return ttest_rel(d1, d2, alternative='greater').pvalue