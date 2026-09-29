import numpy as np
import matplotlib.pyplot as plt

# Newton-Raphson method function
def newton_raphson(f, df, d2f, iterations = 2):
    x = 0
    for i in range(iterations):
        x = x - df(x) / d2f(x)
    return x

def set_up_functions(x0, y0, func_type):
    if func_type == 'parabola':
        y = lambda x: x**2 + 5  # Parabolic function

        f = lambda x: (x-x0)**2 + ((x**2 + 5) - y0)**2  # Distance function
        df = lambda x: 2*(x-x0) + 2*((x**2 + 5) - y0)*(2*x)  # First derivative
        d2f = lambda x: 2 + 2*(2*x)**2 + 2*((x**2 + 5) - y0)*(2)  # Second derivative

    elif func_type == 'exponential':
        y = lambda x: np.exp(x)  # Exponential function

        f = lambda x: (x-x0)**2 + (np.exp(x) - y0)**2  # Distance function
        df = lambda x: 2*(x-x0) + 2*(np.exp(x) - y0)*np.exp(x)  # First derivative
        d2f = lambda x: 2 + 2*(np.exp(x))**2 + 2*(np.exp(x) - y0)*np.exp(x)  # Second derivative

    root = newton_raphson(f, df, d2f)  # Call Newton-Raphson method
    return y, root

def main():
    ### EXAMPLE 1: Distance from point to a parabolic function.
    points = [(0,0), (-4,0), (-8,0), (2,0), (6,0)]  # Point coordinates
    function = 'parabola'
    for x0, y0 in points:
        y, root = set_up_functions(x0, y0, function)
        distance = np.sqrt(root**2 + y(root)**2)
        
        print(f"Estimated root ({function} to point equation) after 2 iterations: {root}")
        print(f"Estimated distance from point ({x0}, {y0}) to {function}: {distance}")

    ### EXAMPLE 2: Distance from point to an exponential function.
    function = 'exponential'  # Function type
    for x0, y0 in points:
        y, root = set_up_functions(x0, y0, function)

        distance = np.sqrt(root**2 + y(root)**2)

        print(f"Estimated root ({function} to point equation) after 2 iterations: {root}")
        print(f"Estimated distance from point ({x0}, {y0}) to {function}: {distance}")

if __name__ == "__main__":
    main()