import numpy as np
import matplotlib.pyplot as plt

def estimate_coef(x, y):
    n = np.size(x)
    m_x = np.mean(x)
    m_y = np.mean(y)
    SS_xy = np.sum(y * x) - n * m_y * m_x
    SS_xx = np.sum(x * x) - n * m_x * m_x
    b1 = SS_xy / SS_xx
    b0 = m_y - b1 * m_x
    return (b0, b1)

def plot_regression_line(x, y, b):
    plt.scatter(x, y, color="m", marker="o", s=30)

    y_pred = b[0] + b[1] * x

    plt.plot(x, y_pred, color="g")

    plt.xlabel('x')
    plt.xlabel('y')
    plt.show()
def main():
    x = np.array([5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6])
    y = np.array([99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86])
    b = estimate_coef(x, y)
    print("Estimated coeefficient:\nb_0 = {}\nb_1 = {}".format(b[0], b[1]))
    plot_regression_line(x, y, b)
if __name__ == "__main__":
    main()
