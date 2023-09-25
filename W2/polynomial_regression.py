import numpy as np

x = np.arange(0,20,1)
y = x**2

# Feature engineering that aims to achieve a polynomial functions
X = np.c_[x, x**2, x**3]