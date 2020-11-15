import math
import numpy as np
import matplotlib.pyplot as plt

w = np.random.randn(1)
X = np.arange(-5, 5, step=0.1)

Y = []
for x in X:
    bias = np.random.rand()
    y = w * x + bias
    Y.append(y.item())

Y = np.asarray(Y)

plt.plot(X, Y)

w = np.random.randn(5)


def fn(x, k=1):
    y = x.copy()
    y[y <= 0] = 0
    y[y > 0] = k / (1 + math.e**(-y[y > 0]/math.e + 3*math.e))
    return y

y = fn(x)
plt.plot(x, y)
