import numpy as np
import matplotlib.pyplot as plt

# Bisection method function
def bisection(df, func_type, tolerance=1e-7, iterations=100):
    x_1, x_2 = (-10,10)
    hist = [(x_1, x_2)]

    for i in range(iterations):
        x_3 = (x_1 + x_2) / 2
        f_3 = df(x_3)

        if f_3 > 0:
            x_2 = x_3
        else:
            x_1 = x_3

        hist.append((x_1, x_2))
        if abs(x_2 - x_1) < tolerance:
            return x_3, hist

    return (x_1 + x_2) / 2, hist

def set_up_functions(x0, y0, func_type):
    if func_type == 'parabola':
        y = lambda x: x**2 + 5  # Parabolic function
        df = lambda x: 2*(x-x0) + 2*((x**2 + 5) - y0)*(2*x)

    elif func_type == 'exponential':
        y = lambda x: np.exp(x)  # Exponential function
        df = lambda x: 2*(x-x0) + 2*(np.exp(x) - y0)*np.exp(x)

    root, history = bisection(df, func_type)
    return y, root, history

def plot_distances(points_list, func_type):
    x_func = np.linspace(-10, 10, 1000)
    y_curve = x_func**2 + 5 if func_type == 'parabola' else np.exp(x_func)
    plt.figure(figsize=(20, 5))
    plt.plot(x_func, y_curve, 'b-', label=func_type)

    for i, (x_0, y_0) in enumerate(points_list):
        colour = plt.cm.tab10(i)
        y, root, history = set_up_functions(x_0, y_0, func_type)
        distance = np.sqrt((root - x_0)**2 + (y(root) - y_0)**2)

        plt.plot([x_0, root], [y_0, y(root)], '-', color=colour)  # Line from point to function
        plt.plot(x_0, y_0, 'o', color=colour)
        for k, (left, right) in enumerate(history):
            alpha = (k + 1) / len(history)  # older intervals are more transparent
            plt.plot([left, right], [y(left), y(right)], '-', color=colour,
                     alpha=alpha, linewidth=1 + 2 * alpha, zorder=2 + k)
            plt.plot(left, y(left), '>', color=colour, alpha=alpha, markersize=8, zorder=2 + k)
            plt.plot(right, y(right), '<', color=colour, alpha=alpha, markersize=8, zorder=2 + k)

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