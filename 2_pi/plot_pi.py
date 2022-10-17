import matplotlib.pyplot as plt
import numpy as np
import time


def pi_calculation(n):
    dn = 1.0 / n
    x = np.arange(0.0, 1.0, dn)
    y = np.sqrt(1.0 - x**2)
    return 4.0 * np.sum(y) * dn


if __name__ == "__main__":
    n = [1000, 10000, 100000, 1000000, 10000000]
    pi = [(pi_calculation(i)-np.pi)/np.pi for i in n]
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(n, pi, 'ro')
    plt.xscale('log')
    plt.xlabel('n')
    plt.yscale('log')
    plt.ylabel('Relative error')

    t = []
    for i in n:
        t0 = time.time()
        pi_calculation(i)
        t1 = time.time()
        t.append(t1-t0)
    plt.subplot(2, 1, 2)
    plt.plot(n, t, 'ro')
    plt.xscale('log')
    plt.xlabel('n')
    plt.ylabel('Time')
    plt.show()
