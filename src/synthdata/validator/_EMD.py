# -*- coding: utf-8 -*-
"""
EMD

@author: david
"""

import numpy as np

def best(dist, move):
    is_best = True
    base = np.arange(move.size)
    for i, j in enumerate(move):
        diff = (dist[base, j] + dist[i, move]) - (dist[i, j] + dist[base, move])
        ind = np.argmin(diff)
        if diff[ind] < 0:
            is_best = False
            move[ind], move[i] = move[i], move[ind]
    return is_best

def EMD(A, B, n_attempts=3, return_movement=False):
    assert A.shape == B.shape, "Size mismatch"
    n, d = A.shape
    dist = np.sqrt(np.sum(np.square(
        np.reshape(B, (1, n, d)) - np.reshape(A, (n, 1, d)
        )), 2))
    emd = None
    movement = None
    for _ in range(n_attempts):
        move = np.random.choice(n, n, False)
        while not best(dist, move): pass
        new_emd = sum([dist[i,j] for i, j in enumerate(move)]) / n
        if emd is None or new_emd < emd:
            emd = new_emd
            movement = move
    if return_movement:
        return emd, movement
    return emd