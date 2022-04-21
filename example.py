import numpy as np
import matplotlib.pyplot as plt
from fft import fft
from ifft import ifft


t = np.linspace(-100, 100, 2 ** 10)
x = np.sinc(t)

X = fft(x)
plt.plot(np.fft.fftshift(abs(X)))
plt.show()

x = ifft(X)
plt.plot(t, x)
plt.show()
