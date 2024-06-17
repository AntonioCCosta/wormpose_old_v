"""
Contains functions to calculate distances between worm poses, either represented as angles or as skeletons.
"""

import math
import numpy as np


def angle_distance(theta_a: np.ndarray, theta_b: np.ndarray) -> float:
    """
    Angle distance that takes into account the periodicity of angles.
    
    :param theta_a: First set of angles.
    :param theta_b: Second set of angles.
    :return: Mean of the absolute differences between the angles.
    """
    diff = np.abs(np.arctan2(np.sin(theta_a - theta_b), np.cos(theta_a - theta_b)))
    return diff.mean()


def _head_tail_diff(skel: np.ndarray) -> Tuple[float, float]:
    """
    Calculate the difference between the head and tail of a skeleton.
    
    :param skel: Skeleton coordinates.
    :return: Difference in x and y coordinates.
    """
    return skel[-1][0] - skel[0][0], skel[-1][1] - skel[0][1]


def _cos_similarity(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """
    Calculate the cosine similarity between two vectors.
    
    :param a: First vector.
    :param b: Second vector.
    :return: Cosine similarity.
    """
    def _norm(x: Tuple[float, float]) -> float:
        return math.sqrt(x[0] * x[0] + x[1] * x[1])

    return (a[0] * b[0] + a[1] * b[1]) / (_norm(a) * _norm(b))


def skeleton_distance(skel_a: np.ndarray, skel_b: np.ndarray) -> float:
    """
    Cosine similarity between the two head-to-tail vectors of the input skeletons.
    
    :param skel_a: First skeleton coordinates.
    :param skel_b: Second skeleton coordinates.
    :return: Cosine similarity.
    """
    return _cos_similarity(_head_tail_diff(skel_a), _head_tail_diff(skel_b))
