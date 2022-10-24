import numpy as np
import matplotlib.pyplot as plt

# Constants
# ---------
# Amplitude
A = 1.0
# Spring constant
k = 1.0
# Mass
m = 1.0
# Stop time
T = 20.0

def dy(t, y):
    '''
    calculate the derivative of y with respect to t
    '''
    return np.array([y[1], -k/m*y[0]])

def euler_method(*arg):
    '''Simple harmonic motion with initial conditions solve by euler method'''
    x0, v0, n = arg
    h = T / n
    y = np.array([x0, v0])
    x = np.zeros(n)
    v = np.zeros(n)
    x[0] = x0
    v[0] = v0
    for i in range(1, n):
        y = y + h*dy(i*h, y)
        x[i] = y[0]
        v[i] = y[1]
    return x, v

def rk2(*arg):
    '''Simple harmonic motion with initial conditions solve by RK2'''
    x0, v0, n = arg
    h = T / n
    y = np.array([x0, v0])
    x = np.zeros(n)
    v = np.zeros(n)
    x[0] = x0
    v[0] = v0
    for i in range(1, n):
        k1 = dy(i*h, y)
        k2 = dy(i*h + h, y + h*k1)
        y = y + h/2*(k1 + k2)
        x[i] = y[0]
        v[i] = y[1]
    return x, v

def rk4(*arg):
    '''Simple harmonic motion with initial conditions solve by RK4'''
    x0, v0, n = arg
    h = T / n
    y = np.array([x0, v0])
    x = np.zeros(n)
    v = np.zeros(n)
    x[0] = x0
    v[0] = v0
    for i in range(1, n):
        k1 = dy(i*h, y)
        k2 = dy(i*h + h/2, y + h/2*k1)
        k3 = dy(i*h + h/2, y + h/2*k2)
        k4 = dy(i*h + h, y + h*k3)
        y = y + h/6*(k1 + 2*k2 + 2*k3 + k4)
        x[i] = y[0]
        v[i] = y[1]
    return x, v

def call():
    '''Call the functions'''
    x0 = A
    v0 = 0.0
    n = 50
    x1, v1 = euler_method(x0, v0, n)
    x2, v2 = rk2(x0, v0, n)
    x3, v3 = rk4(x0, v0, n)
    t = np.linspace(0, 20, n)
    plt.plot(t, np.cos(t), label='Analytical')
    plt.plot(t, x1, label='Euler')
    plt.plot(t, x2, label='RK2')
    plt.plot(t, x3, label='RK4')
    plt.ylim(-1.1, 1.1)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    call()
