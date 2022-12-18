import os
os.chdir(os.path.dirname(os.path.realpath(__file__)) + '/..')
path = os.getcwd() + "/src"
if path not in os.sys.path:
    os.sys.path.append(path)

import numpy as np
import matplotlib.pyplot as plt

from synthdata.generator import GMM

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

gmm = GMM(attempts=5)
gmm.fit(X)

syn = 1000
fdata = np.random.normal(mean, np.sqrt(var) / 2, size=(syn, 2)) + np.full((syn, 2), [np.nan, 0])
F = gmm.fill((fdata - mean) / np.sqrt(var))
S = gmm.generate(syn)

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
probs = gmm.probabilities(xy)
print(np.sum(probs) * delta ** 2)

### BIC per diferents valors de k

dim = X.shape[1]
n = X.shape[0]
probs = []
ks = []
for j in range(10):
    k = j + 1
    gmm._set_k(k)
    gmm.fit(X)
    
    probs.append(
        k * (dim * (dim + 1) / 2 + dim + 1) * np.log(n)
        - 2 * gmm.loglikelihood(X)
        )
    ks.append(k)
    
fig, ax = plt.subplots(1, 1)
ax.plot(ks, probs)
plt.show()