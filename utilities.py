import numpy as np


def arr_modified(arr):
    arr_modified = arr.astype(np.uint16)
    arr_modified1 = arr_modified.copy()
    max_value = arr.max()
    arr_modified[arr_modified > max_value] = 0

    return arr_modified, arr_modified1

