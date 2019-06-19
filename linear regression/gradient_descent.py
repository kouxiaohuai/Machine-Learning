# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 14:18:00 2019

@author: kWX596514
"""

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

# rng = np.random.RandomState(42)
x = 10 * np.random.rand(50)
y = 2 * x - 1 + np.random.randn(50)
plt.scatter(x, y)


def loss_function(theta_0, theta_1):
    mse = 0
    for i in range(len(x)):
        mse += (theta_0 + x[i]*theta_1 - y[i])**2
    mse /= len(x)
    return mse


def gradient_0(theta_0, theta_1):
    gra_0 = 0
    for i in range(len(x)):
        gra_0 += (theta_0 + x[i]*theta_1 - y[i])
    gra_0 *= 2
    gra_0 /= len(x)
    return gra_0


def gradient_1(theta_0, theta_1):
    gra_1 = 0
    for i in range(len(x)):
        gra_1 += x[i] * (theta_0 + x[i]*theta_1 - y[i])
    gra_1 *= 2
    gra_1 /= len(x)
    return gra_1


eta = 0.02
n_iterations = 1000
theta_0 = np.random.randn()
theta_1 = np.random.randn()
lf = loss_function(theta_0, theta_1)
theta_0_list = [theta_0]
theta_1_list = [theta_1]
lf_list = [lf]

for iteration in range(n_iterations):
    theta_0 -= eta * gradient_0(theta_0_list[-1], theta_1_list[-1])
    theta_1 -= eta * gradient_1(theta_0_list[-1], theta_1_list[-1])
    theta_0_list.append(theta_0)
    theta_1_list.append(theta_1)
    lf_list.append(loss_function(theta_0, theta_1))

fig = plt.figure()
ax = plt.axes(projection='3d')

theta_0_area = np.linspace(min(theta_0_list), max(theta_0_list), 1000)
theta_1_area = np.linspace(min(theta_1_list), max(theta_1_list), 1000)
THETA_0, THETA_1 = np.meshgrid(theta_0_area, theta_1_area)
MSE = loss_function(THETA_0, THETA_1)

surf = ax.plot_surface(THETA_0, THETA_1, MSE, cmap=cm.rainbow,
                       linewidth=0, antialiased=False, alpha=0.2)
fig.colorbar(surf, shrink=0.5, aspect=5)

ax.plot3D(theta_0_list, theta_1_list, lf_list)
ax.scatter3D(theta_0_list, theta_1_list, lf_list)
