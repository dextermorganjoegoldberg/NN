# -*- coding:utf-8 -*-
"""
Title:
Author:He Hulingxiao
Date:2022.10.29
"""

import numpy as np
from numpy.random import randn

N,D_in,H,D_out = 64, 1000, 100, 10
x, y = randn(N,D_in), randn(N,D_out)
w1, w2 = randn(D_in, H), randn(H, D_out)

for t in range(2000):
    # h: N x H
    h = 1 / (1 + np.exp(-x.dot(w1))) # sigmoid
    # y_pred: N x D_out
    y_pred = h.dot(w2)
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    # N x D_out
    grad_y_pred = 2.0 * (y_pred - y)
    # H x D_out
    grad_w2 = h.T.dot(grad_y_pred)
    # D_out x D_out
    grad_h = grad_y_pred.dot(w2.T)
    # D_in x H
    grad_w1 = x.T.dot(grad_h * h * (1-h))

    # D_in x H
    w1 -= 1e-4 * grad_w1
    # H x D_out
    w2 -= 1e-4 * grad_w2