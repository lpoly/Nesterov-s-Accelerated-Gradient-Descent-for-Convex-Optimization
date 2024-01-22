# Optimization Algorithms

## Description

This repository contains implementations of various optimization algorithms including Nesterov Accelerated Gradient Descent (NAGD), Gradient Descent, and Gradient Descent with Momentum. Additionally, it features two line search algorithms following Armijo's principles. The code is designed to provide a comprehensive understanding of these algorithms' behaviors on different types of functions.

## Contents

- `optimization_methods.py`: Core implementation of the optimization algorithms.
- `Example1.2`: Implementation of an example from a reference paper, aiming to reproduce the results.
- `Ackley-Styblinski_Tang Plotting`: Graphical representation of the optimization algorithms on non-convex functions such as Ackley and Styblinski-Tang.
- `convex_comparison.py`: A comparison of the algorithms for a convex function in various dimensions.

## Features

- **Flexible Algorithm Parameters**: Wide selection of parameters for each algorithm, including the option to use line search, choice of line search algorithm, stopping criteria, step size (constant or reducing), maximum iterations, and tolerance level.
- **Line Search Algorithms**: Implementations based on Armijo's principles to dynamically adjust step sizes.
- **Comparative Analysis**: Tools for comparing algorithm performance on convex and non-convex functions.

## Usage

This repository provides implementations of various optimization algorithms, including Nesterov Accelerated Gradient Descent (NAGD), Gradient Descent, and Gradient Descent with Momentum, along with line search algorithms following Armijo's principles.


## Example Usage

```python
import numpy as np
from your_module_name import gradient_descent, gradient_descent_with_momentum, nesterov_momentum

# Define your objective function and its derivative
def func(x):
    # Your function implementation
    pass

def dfunc(x):
    # Derivative of your function
    pass

# Example configuration and usage of Gradient Descent
gd_result = gradient_descent(func, dfunc, lr=0.1, x_init=np.array([1, 1]), max_iter=1000, tol=1e-6, line_search=True)

# Example configuration and usage of Gradient Descent with Momentum
gdm_result = gradient_descent_with_momentum(func, dfunc, beta=0.9, lr=0.1, x_init=np.array([1, 1]), max_iter=5000, tol=1e-6, line_search=False)

# Example configuration and usage of Nesterov Accelerated Gradient Descent
nagd_result = nesterov_momentum(func, dfunc, t=0.1, lamda=0.9, x_init=np.array([1, 1]), max_iter=4000, tol=1e-6, stop_criterion=0, line_search=True, dec_stepsize=True)


## Requirements

Make sure to have NumPy installed in your environment, as these examples use `numpy.array` for initializing `x_init`.
```


## Running the Examples

To run the examples, ensure that you have the Python environment set up with necessary dependencies (like NumPy for numerical operations) and simply execute the Python scripts.

## Additional Notes

- Customize the parameters in the example scripts according to the specific needs of the optimization problem you are solving.
- The line search feature can be toggled on or off based on the requirement of your optimization task.


