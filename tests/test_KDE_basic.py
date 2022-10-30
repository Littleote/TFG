import os
os.chdir(os.path.dirname(os.path.realpath(__file__)) + '/..')
path = os.getcwd() + "/src"
if path not in os.sys.path:
    os.sys.path.append(path)

import numpy as np
import matplotlib.pyplot as plt

from synthdata.generator import KDE

n = 100

phase = np.linspace(0, 2 * np.pi, n)
noise = [np.random.normal(size=n) for _ in range(2)]
data = np.array([
        np.sin(phase + noise[0]) + .1 * np.sin(phase + noise[1]),
        np.cos(phase + noise[0]) + .1 * np.cos(phase + noise[1])
    ]).transpose() * 10 + np.array([1, 5])

mean = np.mean(data, 0)
var = np.var(data, 0)

X = (data - mean) / np.sqrt(var)

### fit, generate i fill test

kde = KDE()
kde.fit(X)

syn = 1000
fdata = np.random.normal(mean, np.sqrt(var) / 2, size=(syn, 2)) + np.full((syn, 2), [np.nan, 0])
F = kde.fill((fdata - mean) / np.sqrt(var))
S = kde.generate(syn)

sdata = S * np.sqrt(var) + mean
fdata = F * np.sqrt(var) + mean

fig, ax = plt.subplots(1, 1)
sp = ax.scatter(sdata[:,0], sdata[:,1], c='yellow', alpha=.2)
sp = ax.scatter(fdata[:,0], fdata[:,1], c='green', alpha=.1)
sp = ax.scatter(data[:,0], data[:,1], c='red', alpha=.2)

plt.show()

### Probabilitat total ~= 1

steps = 200
size = 2
delta = 2 * size / steps
space = np.linspace(-size, size, steps)
x, y = np.meshgrid(space, space)
xy = np.reshape(np.stack([x, y], 2), (-1, 2))
probs = kde.probabilities(xy)
print(np.sum(probs) * delta ** 2)
