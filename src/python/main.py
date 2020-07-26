from artifacts import _simple_petsc_example
from matplotlib import pyplot as plt
import numpy as np

def main():
    initial_guesses = [
        [-3.0, 0.0],
        [6.0, 6.0],
        [1.0, 1.0],
        [-1.0, -2.5],
    ]

    xlist = np.linspace(-7.5, 7.5, 60)
    ylist = np.linspace(-7.5, 7.5, 60)
    X, Y = np.meshgrid(xlist, ylist)

    Z = X**2 + Y**2
    plt.contour(
        X,
        Y,
        Z,
        levels=[20],
        colors=('b',),
        linestyles=('--',),
        linewidths=(1,)
    )
    plt.contourf(X, Y, Z, alpha=.75)

    Z = X - Y
    plt.contour(
        X,
        Y,
        Z,
        levels=[-2],
        colors=('b',),
        linestyles=('--',),
        linewidths=(1,),
    )
    plt.contourf(X, Y, Z, alpha=.25)

    colors = ['y', 'r', 'w', 'c']
    for c, (xi, yi) in zip(colors, initial_guesses):
        steps_info = _simple_petsc_example.run_solver(xi, yi)
        steps_array = np.array(steps_info)
        plt.plot(steps_array[:, 0], steps_array[:, 1], f'{c}-o')

    plt.show()

if __name__ == "__main__":
    try:
        _simple_petsc_example.init_petsc()
        main()
    finally:
        _simple_petsc_example.finalize_petsc()
