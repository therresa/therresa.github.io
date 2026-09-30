import numpy as np
import matplotlib.pyplot as plt

def newton_raphson(params, pts_list, func_type, tolerance=1e-7, iterations=100):
    hist = [tuple(params)]
    for i in range(iterations):
        old_params = params
        for j in range(len(params)):
            # recompute the derivatives at the latest params so each update sees the newest values of the others
            # where first, second correspond to the first and second derivatives of the MSE
            first, second = set_up_derivatives(pts_list, params, func_type)
            new_params = list(params)
            # guarding against particularly small or negative second derivatives (Newton-Raphson divergence cases)
            if second[j] < 1e-12:
                new_params[j] = params[j] - 0.1 * first[j]
            else:
                new_params[j] = params[j] - first[j] / second[j]
            params = tuple(new_params)
        hist.append(params)
        if max(abs(a - b) for a, b in zip(params, old_params)) < tolerance:
            break
    plot_fitted_func(pts_list, params, func_type, hist)
    return params, hist

def set_up_derivatives(pts_list, params, func_type):
    if func_type == 'parabola':
        B, C, D = params
        df_dB = df_dC = df_dD = 0.0
        df_d2B = df_d2C = df_d2D = 0.0
        for x, y in pts_list:
            f = B*x**2 + C*x + D - y
            df_dB += 2 * f * x**2
            df_d2B += 2 * x**2 * x**2
            df_dC += 2 * f * x
            df_d2C += 2 * x * x
            df_dD += 2 * f
            df_d2D += 2
        return [df_dB, df_dC, df_dD], [df_d2B, df_d2C, df_d2D]
    if func_type == 'linear':
        C_0, C_1 = params
        df_dC_0 = df_dC_1 = 0.0
        df_d2C_0 = df_d2C_1 = 0.0
        for x, y in pts_list:
            f = C_0 + C_1 * x - y
            df_dC_0 += 2 * f
            df_d2C_0 += 2
            df_dC_1 += 2 * f * x
            df_d2C_1 += 2 * x * x
        return [df_dC_0, df_dC_1], [df_d2C_0, df_d2C_1]

def plot_fitted_func(points_list, params, func_type, history):
    x_pts = np.linspace(-10, 10, 1000)
    if func_type == 'parabola':
        B, C, D = params
        y_curve = B*x_pts**2 + C*x_pts + D
        plt.plot(x_pts, y_curve, 'b-', label='{B:.3f}x^2 + {C:.3f}x + {D:.3f}'.format(B=B, C=C, D=D))
    elif func_type == 'linear':
        C_0, C_1 = params
        y_curve = C_0 + C_1 * x_pts
        plt.plot(x_pts, y_curve, 'r-', label='{C_0:.3f} + {C_1:.3f}x'.format(C_0=C_0, C_1=C_1))

    for i, (x_0, y_0) in enumerate(points_list):
        colour = plt.cm.tab10(i)
        plt.plot(x_0, y_0, 'o', color=colour)
        plt.text(x_0, y_0, f'({x_0}, {y_0})', fontsize=9, ha='left')

    if func_type == 'parabola':
        for k, param_guess in list(enumerate(history))[::20]:
            B_guess, C_guess, D_guess = param_guess
            y_guess = B_guess*x_pts**2 + C_guess*x_pts + D_guess
            plt.plot(x_pts, y_guess, '--', color='gray', alpha=(k + 1) / len(history))
    elif func_type == 'linear':
        for k, param_guess in enumerate(history):
            C_0_guess, C_1_guess = param_guess
            y_guess = C_0_guess + C_1_guess * x_pts
            plt.plot(x_pts, y_guess, '--', color='gray', alpha=(k + 1) / len(history))

    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.title(f'Fitted {func_type} function to points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

def main():
    ### EXAMPLE 1: Fit a parabola to the points.
    points = [(0, 0.5), (2, 3.5), (1, 1.5), (3, 7.5)]
    function = 'parabola'

    params, hist = newton_raphson((0, 0, 0), points, function, tolerance=1e-7, iterations=100)
    print(function, params, f"({len(hist) - 1} sweeps)")

    ### EXAMPLE 2: Fit a line to the points.
    function = 'linear'

    params, hist = newton_raphson((0, 0), points, function, tolerance=1e-7, iterations=100)
    print(function, params, f"({len(hist) - 1} sweeps)")

if __name__ == "__main__":
    main()