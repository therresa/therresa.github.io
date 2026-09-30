import numpy as np
import matplotlib.pyplot as plt

def newton_raphson(f, df, d2f, tolerance=1e-7, iterations=100):
    x = 0
    hist = [x]
    for i in range(iterations):
        if d2f(x) < 1e-12: 
             x = x - 0.1 * df(x)
        else:
            x = x - df(x) / d2f(x)
        if abs(x - hist[-1]) < tolerance:
            break
    hist.append(x)
    return x, hist

def set_up_functions(x0, y0, func_type):
    if func_type == 'parabola':
        y = lambda x: x**2 + 5

        f = lambda x: (x-x0)**2 + ((x**2 + 5) - y0)**2
        df = lambda x: 2*(x-x0) + 2*((x**2 + 5) - y0)*(2*x) 
        d2f = lambda x: 2 + 2*(2*x)**2 + 2*((x**2 + 5) - y0)*(2) 

    elif func_type == 'exponential':
        y = lambda x: np.exp(x)

        f = lambda x: (x-x0)**2 + (np.exp(x) - y0)**2 
        df = lambda x: 2*(x-x0) + 2*(np.exp(x) - y0)*np.exp(x)
        d2f = lambda x: 2 + 2*(np.exp(x))**2 + 2*(np.exp(x) - y0)*np.exp(x) 

    root, history = newton_raphson(f, df, d2f)
    return y, root, history

def plot_distances(points_list, func_type):
    x_pts = np.linspace(-10, 10, 1000)
    y_curve, root, history = set_up_functions(0, 0, func_type)
    y_curve = y_curve(x_pts)
    plt.figure(figsize=(20, 5))
    plt.plot(x_pts, y_curve, 'b-', label=func_type)

    for i, (x_0, y_0) in enumerate(points_list):
        colour = plt.cm.tab10(i)
        y, root, history = set_up_functions(x_0, y_0, func_type)
        distance = np.sqrt((root - x_0)**2 + (y(root) - y_0)**2)

        plt.plot([x_0, root], [y_0, y(root)], '-', color=colour)
        plt.plot(x_0, y_0, 'o', color=colour)
        for k, x_guess in enumerate(history):
            # fading alpha based on when the guess was made during the iterations
            plt.plot(x_guess, y(x_guess), 'o', color=colour, alpha=(k + 1) / len(history), markersize=8)

        plt.text(x_0, y_0, f'({x_0}, {y_0})', fontsize=9, ha='left')
        plt.text(root, y(root), f'{distance:.2f}', fontsize=9, ha='right', clip_on=True)

    plt.xlim(-10, 10)
    plt.ylim(-1, 20)
    plt.title(f'Distance from points to {func_type}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid()
    plt.show()

def main():
    ### EXAMPLE 1: Distance from point to a parabolic function.
    points = [(0,0), (-4,0), (-8,0), (2,0), (6,0), (4,6)]
    function = 'parabola'

    plot_distances(points, function)

    ### EXAMPLE 2: Distance from point to an exponential function.
    function = 'exponential' 

    plot_distances(points, function)

if __name__ == "__main__":
    main()