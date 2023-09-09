import numpy as np


def fft(x):
    """Returns the discrete Fourier transform of a sequence.

    Args:
        x: A sequence of complex numbers

    Returns:
        The discrete Fourier transform of the sequence
    """
    n = len(x)
    if n <= 1:
        return x

    X0 = fft(x[0::2])
    X1 = fft(x[1::2])
    X = np.concatenate((X0, X1))

    A = np.zeros((n, n), dtype=complex)
    m = n // 2
    e = np.exp(-1j * 2 * np.pi / n)
    for i in range(n):
        j = i % m
        A[i][j    ] = 1
        A[i][j + m] = e ** i

    return np.dot(A, X) / 2 ** 0.5
