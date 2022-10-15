import os
os.chdir(os.path.dirname(os.path.realpath(__file__)) + '/..')
os.sys.path.append(os.getcwd() + "/src")

import numpy as np
import matplotlib.pyplot as plt

from synthdata.validator import EMD

n = 200

### Test with two normals

d = 2
A = np.random.normal(size=(n, d))
B = np.random.normal(size=(n, d))

emds = [EMD(A, B, n_attempts=1) for i in range(20)]
plt.plot(emds, c='red')
plt.plot(np.minimum.accumulate(emds), c='blue')
plt.title("Earth movers distance precision increase with iterations")
plt.show()

def plot_movement(A, B, move, name):
    fig, ax = plt.subplots()
    
    ax.scatter(*A.T[:2], c='red')
    ax.scatter(*B.T[:2], c='green')
    ax.quiver(*A.T[:2], *(B[move] - A).T[:2], color='blue', angles='xy', scale_units='xy', scale=1)
    
    plt.title(name)
    plt.show()
    
plot_movement(A, B, EMD(A, B, return_movement=True)[1], "normal - normal")

### Test with non-normal distributions

noise = np.random.normal((0, 1), (1, .2), (n, 2)).T
A = np.array([noise[1] * np.cos(noise[0]), noise[1] * np.sin(noise[0])]).T

noise = np.random.normal((0, 1), (1, .2), (n, 2)).T
A_prime = np.array([noise[1] * np.cos(noise[0]), noise[1] * np.sin(noise[0])]).T

B = np.random.normal(np.mean(A), np.sqrt(np.var(A)), A.shape)

print("A to A_prime:", EMD(A, A_prime))
plot_movement(A, A_prime, EMD(A, A_prime, return_movement=True)[1], "A to A_prime")
print("A to B:", EMD(A, B))
plot_movement(A, B, EMD(A, B, return_movement=True)[1], "A to B")