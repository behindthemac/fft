import numpy as np


def dft(x):
    n = len(x)
    e = np.exp(-1j * 2 * np.pi / n)
    A = [[i * j for j in range(n)] for i in range(n)]
    return np.dot(e ** A, x) / n ** 0.5
