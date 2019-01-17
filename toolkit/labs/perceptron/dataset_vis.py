import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # greater-than
    d1x_true = np.array([0.1, -0.55, -0.65, 0.8])
    d1y_true = np.array([0.2, 0.56, -0.35, 0.9])
    d1x_false = np.array([0.6, 0.13, -0.089, 0.9])
    d1y_false = np.array([0.5, 0.01, -0.29, -0.8])

    # all-positive
    d2x_true = np.array([0.4, 0.1, 0.95, 0.08])
    d2y_true = np.array([0.2, 0.1, 0.14, 0.67])
    d2x_false = np.array([-0.4, -0.8, -0.1, 0.86])
    d2y_false = np.array([-0.2, 0.1, 0.905, -0.09])

    plt.title("Greater Than (Linearly Separable)")
    plt.plot(d1x_true, d1y_true, "go", label="y > x")
    plt.plot(d1x_false, d1y_false, "rx", label="y <= x")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.grid(True)
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.show()

    plt.title("All Positive (Linearly Inseperable)")
    plt.plot(d2x_true, d2y_true, "go", label="both positive")
    plt.plot(d2x_false, d2y_false, "rx", label="either negative")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.grid(True)
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.show()
