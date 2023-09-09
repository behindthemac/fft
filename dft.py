import numpy as np


def dft(x):
    """Returns the discrete Fourier transform of a sequence.

    Args:
        x: A sequence of complex numbers

    Returns:
        The discrete Fourier transform of the sequence
    """
    n = len(x)
    e = np.exp(-1j * 2 * np.pi / n)
    A = [[i * j for j in range(n)] for i in range(n)]
    return np.dot(e ** A, x) / n ** 0.5
