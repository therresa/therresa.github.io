import numpy as np
import matplotlib.pyplot as plt

# Newton-Raphson method function
def newton_raphson(f, df, d2f, iterations = 10):
    x = 0
    hist = [x]
    for i in range(iterations):
        if d2f(x) < 1e-12: 
             x = x - 0.1 * df(x)
        else:
            x = x - df(x) / d2f(x)
    hist.append(x)
    return x, hist

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

    root, history = newton_raphson(f, df, d2f)  # Call Newton-Raphson method
    return y, root, history

def plot_distances(points_list, func_type, history):
    zoom = func_type == 'parabola'
    if zoom:
        fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(20, 5), gridspec_kw={'width_ratios': [2, 1]})
        axes = (ax_full, ax_zoom)
        label_ax = ax_zoom
    else:
        fig, ax_full = plt.subplots(figsize=(20, 5))
        axes = (ax_full,)
        label_ax = ax_full

    x_func = np.linspace(-10, 10, 1000)
    y = x_func**2 + 5 if func_type == 'parabola' else np.exp(x_func)

    guess_x, guess_y = [], []

    for ax in axes:
        ax.plot(x_func, y, 'b-', label=func_type)

    for i, (x_0, y_0) in enumerate(points_list):
        colour = plt.cm.tab10(i)
        y, root, history = set_up_functions(x_0, y_0, func_type)
        distance = np.sqrt((root - x_0)**2 + (y(root) - y_0)**2)

        for ax in axes:
            ax.plot([x_0, root], [y_0, y(root)], '-', color=colour)  # Line from point to function
            ax.plot(x_0, y_0, 'o', color=colour)
            for k, x_guess in enumerate(history):
                ax.plot(x_guess, y(x_guess), 'o', color=colour, alpha=(k + 1) / len(history), markersize=8)

        guess_x += list(history) + [root]
        guess_y += [y(x_guess) for x_guess in history] + [y(root)]

        ax_full.text(x_0, y_0, f'({x_0}, {y_0})', fontsize=9, ha='left')
        ax_full.text(root, y(root), f'{distance:.2f}', fontsize=9, ha='right', clip_on=True)
        label_ax.text(root, y(root), f'{distance:.2f}', fontsize=9, ha='right', clip_on=True)

    ax_full.set_xlim(-10, 10)
    ax_full.set_ylim(-1, 20)

    if zoom:
        # Zoomed view: limits come from the guesses only
        x_pad = 0.15 * (max(guess_x) - min(guess_x) + 1e-9)
        y_pad = 0.15 * (max(guess_y) - min(guess_y) + 1e-9)
        ax_zoom.set_xlim(min(guess_x) - x_pad, max(guess_x) + x_pad)
        ax_zoom.set_ylim(min(guess_y) - y_pad, max(guess_y) + y_pad)
        ax_zoom.set_title('Zoom on Newton-Raphson guesses')

    ax_full.set_title(f'Distance from points to {func_type}')
    for ax in axes:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()
        ax.grid()
    plt.show()

def main():
    ### EXAMPLE 1: Distance from point to a parabolic function.
    points = [(0,0), (-4,0), (-8,0), (2,0), (6,0), (4,6)]  # Point coordinates
    function = 'parabola'
    for x0, y0 in points:
        y, root, history = set_up_functions(x0, y0, function)
        distance = np.sqrt((root - x0)**2 + (y(root) - y0)**2)
        
        print(f"Estimated root ({function} to point equation) after 10 iterations: {root}")
        print(f"Estimated distance from point ({x0}, {y0}) to {function}: {distance}")

    plot_distances(points, function, history)

    ### EXAMPLE 2: Distance from point to an exponential function.
    function = 'exponential'  # Function type
    for x0, y0 in points:
        y, root, history = set_up_functions(x0, y0, function)

        distance = np.sqrt((root - x0)**2 + (y(root) - y0)**2)

        print(f"Estimated root ({function} to point equation) after 10 iterations: {root}")
        print(f"Estimated distance from point ({x0}, {y0}) to {function}: {distance}")

    plot_distances(points, function, history)

if __name__ == "__main__":
    main()