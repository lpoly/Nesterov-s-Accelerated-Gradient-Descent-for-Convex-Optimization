# -*- coding: utf-8 -*-
"""
For the purposes of MAM T5.02 - Numerical Methods for Optimization
University of Crete - Mathematics and Applied Mathematics Department
Winter 2023

@author: Lefteris Polychronakis
@ID:     math6090

@under professor: Panagiotis Chatzipantelidis

@brief: This file contains the algorithms of some methods for Optimization,
        namely: Nesterov Accelerated Gradient Descent with non fixed param,
                Nesterov Accelerated Gradient Descent with fixed param,
                Gradient Descent with Momentum, Vanilla Gradient Descent
        They're intended to be used for plotting.
"""

import numpy as np


def armijo_nesterov(func, dfunc, x, y, a_init = None):
    a = 1
    if a_init is not None:
        a = a_init

    dfnorm = np.linalg.norm(dfunc(x)) ** 2
    fx = func(x)
    l = fx + a/2  * dfnorm
    cond = func(y - a * dfunc(y)) > l
    ctr = 0
    
    while cond:
        a = a / 2
        # print(a)
        l = fx - a/2 * dfnorm
        cond = func(y - a * dfunc(y)) > l
        
        # loop control
        ctr += 1
        if cond == False:
            # print()
            ctr = 0
        if ctr > 99 or a < 1e-32:
            return None
        
    return a

def nesterov_momentum(func, dfunc, t, lamda = None, x_init = None,
                      line_search = True, max_iter = 1000, tol = 1e-6):
    if x_init is None:
        x_init = np.random.rand(len(x_init))
        
    lamda_i = lamda_i1 = 0
    x_i = x_i1 = y_i = x_init
    x_history = [x_i]

    for i in range(max_iter):
        if line_search :#and i > 0:
            t = armijo_nesterov(func, dfunc, x_i, y_i, a_init=t)    
            if t == None:
                grad_norm = np.linalg.norm(dfunc(x_i1))
                print("Line Search did not terminate. Parameter Underflow encountered")
                print(f"Nesterov terminated after {i+1} iterations:")
                print(f"Min value: {func(x_i1)} Gradient Norm: {grad_norm}\n")
                return x_history

        x_i1 = y_i - t * dfunc(y_i)
        if lamda is None:
            lamda_i1 = (1 + np.sqrt(1 + 4 * lamda_i**2)) / 2
            y_i1 = x_i1 + ((lamda_i - 1) / lamda_i1) * (x_i1 - x_i)
        else:
            y_i1 = x_i1 + (lamda - 1) / (lamda + 1) * (x_i1 - x_i)

        x_history.append(x_i1)

        grad_norm = np.linalg.norm(dfunc(x_i1))
        if grad_norm < tol:
            if not line_search:
                print(f"Nesterov w/o line search Converged after {i+1} iterations:")
                print(f"Min value: {func(x_i1)} Gradient Norm: {grad_norm}\n")
                if lamda is not None:
                    print(f"Nesterov w/o line search and const. param. Converged after {i+1} iterations:")
                    print(f"Min value: {func(x_i1)} Gradient Norm: {grad_norm}\n")
                    
            else:
                if lamda is not None:
                    print(f"Nesterov Converged after {i+1} iterations:")
                    print(f"Min value: {func(x_i1)} Gradient Norm: {grad_norm}\n")
                    
            return x_history

        x_i = x_i1
        y_i = y_i1
        lamda_i = lamda_i1
        
    return x_history


def gradient_descent(func, dfunc, lr = 0.1, x_init = None,
                     max_iter = 1000, tol = 1e-6):
    if x_init is None:
        x_init = np.random.rand(len(x_init))

    x_i = x_init
    x_history = [x_i]

    for i in range(max_iter):
        x_i1 = x_i - lr * dfunc(x_i)

        x_history.append(x_i1)

        grad_norm = np.linalg.norm(dfunc(x_i1))
        if grad_norm < tol:
            print(f"Vanilla Gradient Descent Converged after {i+1} iterations:")
            print(f"Min value: {func(x_i1)} Gradient Norm: {grad_norm}\n")
            return x_history

        x_i = x_i1

    return x_history

def gradient_descent_with_momentum(func, dfunc, beta = 0.9, lr = 0.1,
                                   x_init = None, max_iter = 1000, tol = 1e-6):
    if x_init is None:
        x_init = np.random.rand(len(x_init))

    x_i = x_init
    v_i = np.zeros_like(x_i)
    x_history = [x_i]

    for i in range(max_iter):
        v_i = beta * v_i + lr * dfunc(x_i)
        x_i1 = x_i - v_i

        x_history.append(x_i1)

        grad_norm = np.linalg.norm(dfunc(x_i1))
        if grad_norm < tol:
            print(f"Gradient Descent with Momentum Converged after {i+1} iterations:")
            print(f"Min value: {func(x_i1)} Gradient Norm: {grad_norm}\n")
            return x_history

        x_i = x_i1

    return x_history