# -*- coding: utf-8 -*-
"""
Created on Thu May 30 16:12:14 2019

@author: kWX596514
"""

import math
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
iris = np.hstack((iris.data, iris.target[:, np.newaxis]))


def split_iris(index, iris_sort):
    iris_l = iris_sort[:index, :]
    iris_r = iris_sort[index:, :]
    return iris_l, iris_r


def value_lr(iris_l, iris_r):
    np_l_list = iris_l[:, 4].tolist()
    np_r_list = iris_r[:, 4].tolist()
    value_l = [np_l_list.count(0.0),
               np_l_list.count(1.0),
               np_l_list.count(2.0)]
    value_r = [np_r_list.count(0.0),
               np_r_list.count(1.0),
               np_r_list.count(2.0)]
    return value_l, value_r


def entropy_lr(value_l, value_r):
    value_sum = sum(value_l) + sum(value_r)
    p_l = [value_l[i]/sum(value_l) for i in range(len(value_l))]
    p_r = [value_r[i]/sum(value_r) for i in range(len(value_r))]
    e_l = -sum([p_l[i]*math.log2(p_l[i]) if p_l[i] != 0 else 0 for i in range(len(p_l))])
    e_r = -sum([p_r[i]*math.log2(p_r[i]) if p_r[i] != 0 else 0 for i in range(len(p_r))])
    return sum(value_l)/value_sum*e_l + sum(value_r)/value_sum*e_r


for i in range(4):
    arg = np.argsort(iris[:, i])
    iris_sort = iris[arg]
    
    x = iris_sort[:, i]
    y = []
    
    for j in range(1, len(x)):
        iris_l, iris_r = split_iris(j, iris_sort)
        value_l, value_r = value_lr(iris_l, iris_r)
        ig = -math.log2(1/3) - entropy_lr(value_l, value_r)
        y.append(ig)
    
    plt.grid(True)
    plt.plot(x[1:], y)


plt.legend(labels=[i for i in range(4)], loc=1, fontsize='24')
