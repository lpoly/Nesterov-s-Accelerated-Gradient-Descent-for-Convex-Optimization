# -*- coding: utf-8 -*-
"""
For the purposes of MAM T5.02 - Numerical Methods for Optimization
University of Crete - Mathematics and Applied Mathematics Department
Winter 2023

@author: Lefteris Polychronakis
@ID:     math6090

@under professor: Panagiotis Chatzipantelidis

@brief: This file is used for comparing various optimization algorithms on
        rotated hyper ellipsoid function which is a convex function
"""

import numpy as np
import matplotlib.pyplot as plt
import optimization_methods as tools

def rotated_hyper_ellipsoid(x):
    return np.sum(np.cumsum(x**2, axis=0))

def rotated_hyper_ellipsoid_gradient(x):
    n = len(x)
    return 2 * np.arange(1, n + 1) * x

initial_guess = 3 * np.ones(50,)

nesterov_optimization_history = tools.nesterov_momentum(
    rotated_hyper_ellipsoid,
    rotated_hyper_ellipsoid_gradient,
    x_init = initial_guess,
    t = 0.05
)

# const_param_optimization_history = tools.nesterov_momentum(
#     rotated_hyper_ellipsoid,
#     rotated_hyper_ellipsoid_gradient,
#     x_init = initial_guess,
#     t = 0.05,
#     lamda = 1,
#     # line_search = False,
# )

# wo_ls_optimization_history = tools.nesterov_momentum(
#     rotated_hyper_ellipsoid,
#     rotated_hyper_ellipsoid_gradient,
#     x_init = initial_guess,
#     t = 0.,
#     line_search = False
# )

# momentum_optimization_history = tools.gradient_descent_with_momentum(
#     rotated_hyper_ellipsoid,
#     rotated_hyper_ellipsoid_gradient,
#     x_init = initial_guess,
#     beta=0.85,
#     lr=0.025
# )
