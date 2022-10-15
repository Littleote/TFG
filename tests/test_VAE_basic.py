import os
os.chdir(os.path.dirname(os.path.realpath(__file__)) + '/..')
os.sys.path.append(os.getcwd() + "/src")

import numpy as np
import torch
import matplotlib.pyplot as plt

from synthdata.generator import VAE

n = 500

phase = np.linspace(0, 2 * np.pi, n)
noise = [np.random.normal(size=n) for _ in range(2)]
data = np.array([
        np.sin(phase + noise[0]) + .1 * np.sin(phase + noise[1]),
        np.cos(phase + noise[0]) + .1 * np.cos(phase + noise[1])
    ]).transpose() * 10 + np.array([1, 5])

mean = np.mean(data, 0)
var = np.var(data, 0)

X = (data - mean) / np.sqrt(var)

vae = VAE()
vae.fit(X, layers=4, enc_dim=3, epochs=500, lr=1e-1, batch_size=256)

syn = 500
S = vae.generate(syn)
with torch.no_grad():
    T, M, V = vae.forward(torch.from_numpy(X).float())

sdata = S * np.sqrt(var) + mean
tdata = T.numpy() * np.sqrt(var) + mean

fig, ax = plt.subplots(1, 1)
sp = ax.scatter(sdata[:,0], sdata[:,1], c='yellow', alpha=.2)
# sp = ax.scatter(tdata[:,0], tdata[:,1], c='blue', alpha=.1)
sp = ax.scatter(data[:,0], data[:,1], c='red', alpha=.1)

plt.show()

M = M.numpy()
V = np.exp(V.numpy())

D = (tdata - data) / np.sqrt(var)
plt.scatter(D[:,0], D[:,1], c='brown', alpha=.1)
plt.show()
plt.scatter(M[:,0], M[:,1], c='blue', alpha=.1)
plt.show()
plt.scatter(V[:,0], V[:,1], c='black', alpha=.2)
plt.show()