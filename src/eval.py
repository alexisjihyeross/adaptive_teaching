import numpy as np

""" Helper functions for evaluation """
# TODO: move other functions here


def get_auc(values, target_length=None):
    """
    Extends the gold_total_prob to the max_x (num_iterations) and returns the mean.
    """

    assert target_length >= len(values), f"{target_length} < {len(values)}"

    # if target_length is not None, extend the values to the target length by repeating the last value
    if target_length is not None:
        while len(values) < target_length:
            values.append(values[-1])

    assert len(values) == target_length

    return np.mean(values)


def compute_accuracy(y_true, y_pred):
    # assumes y_true, y_pred are lists
    # just checks equality
    assert len(y_true) == len(y_pred)
    correct = sum([t == p for (t, p) in zip(y_true, y_pred)])
    total = len(y_true)
    if total == 0:
        return np.nan
    return correct / total
