# -*- coding: utf-8 -*-
"""
@author: david
"""

import numpy as np
import pandas as pd

# Sphere dataset
def sphere(N, dimension):
    normal = np.random.normal(size=(N, dimension))
    norm = np.linalg.norm(normal, axis=1, keepdims=True)
    radial_noise = 1 - np.log(np.random.random((N, 1)) + .5) / 10
    vec = normal / norm * radial_noise
    
    sphere = pd.DataFrame(vec, columns=[f'dimension_{i + 1}' for i in range(dimension)])
    sphere['zone'] = ['inner' if r < 1 else 'outter' for r in radial_noise[:,0]]
    return sphere
    
if __name__ == "__main__":
    print(sphere(10, 4))