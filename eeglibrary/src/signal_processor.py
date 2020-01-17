import numpy as np


def to_correlation_matrix(waves):
    return np.corrcoef(waves)


def calc_eigen_values_sorted(matrix):
    if np.any(np.isnan(matrix)):
        exit(1)
    w, v = np.linalg.eig(matrix)
    w = np.absolute(w)
    w.sort()
    return w


# Take the upper right triangle of a matrix
def flatten_corr_upper_right(matrix):
    accum = []
    for i in range(matrix.shape[0]):
        for j in range(i+1, matrix.shape[1]):
            accum.append(matrix[i, j])

    return np.array(accum)
