"""
Definition of the loss function for the network
"""

import tensorflow as tf
import numpy as np

def angle_diff(a, b):
    """
    Root Mean Square Error of the angle difference.
    The angle difference function takes into account the periodicity of angles
    """
    diff = tf.atan2(tf.sin(a - b), tf.cos(a - b))
    return tf.sqrt(tf.reduce_mean(tf.square(diff), axis=1))


def symmetric_angle_difference(y_true, y_pred):
    """
    We calculate the angle difference between the prediction and the two possible labels,
    and pick the minimum of the two,
    we average the result on the batch
    """

    dists = [angle_diff(y_pred, y_true[:, 0]), angle_diff(y_pred, y_true[:, 1])]
    mins = tf.reduce_min(dists, axis=0)
    loss = tf.reduce_mean(mins)
    return loss

# Example usage:
if __name__ == "__main__":
    # Example of computing the loss
    y_true = np.array([[0.5, 0.6], [1.0, 1.1], [2.0, 2.1]])  # Example true labels
    y_pred = np.array([0.45, 0.55, 2.2])  # Example predicted values

    y_true_tf = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred_tf = tf.convert_to_tensor(y_pred, dtype=tf.float32)

    loss_value = symmetric_angle_difference(y_true_tf, y_pred_tf)
    print("Loss:", loss_value.numpy())
