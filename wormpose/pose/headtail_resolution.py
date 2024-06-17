"""
Utility functions to deal with eigenworms.
"""

import numpy as np


def load_eigenworms_matrix(eigenworms_matrix_path: str) -> np.ndarray:
    """
    Load eigenworms matrix into numpy array from a CSV file.

    :param eigenworms_matrix_path: Path of the CSV file.
    :return: Numpy array of the eigenworms matrix.
    """
    return np.loadtxt(eigenworms_matrix_path, delimiter=",").astype(float) if eigenworms_matrix_path else None


def theta_to_modes(theta: np.ndarray, eigenworms_matrix: np.ndarray) -> np.ndarray:
    """
    Convert angles to modes using an eigenworms matrix. The mean angle is subtracted before converting.

    :param theta: Angle vector, numpy array of shape (N,).
    :param eigenworms_matrix: Eigenworms matrix.
    :return: The modes corresponding to the angles.
    """
    return (theta - np.mean(theta)).dot(eigenworms_matrix)


def modes_to_theta(modes: np.ndarray, eigenworms_matrix: np.ndarray) -> np.ndarray:
    """
    Convert modes to angles using an eigenworms matrix.

    :param modes: Modes vector.
    :param eigenworms_matrix: Eigenworms matrix.
    :return: The angles corresponding to the modes.
    """
    return modes.dot(eigenworms_matrix[:, :len(modes)].T)
