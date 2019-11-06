# Minimize the function
# f(x, y) = (1-x)^2 + 100 * (y-x^2)^2
# f(x, y) = (x^2+y-11)^2 + (x+y^2-7)^2
import math
import numpy as np
import matplotlib.pyplot as plt
from sympy import diff, symbols, solve


def func(x, y):  # Rosenbrock
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def cal_x_der(a, b):
    x, y = symbols('x y', real=True)
    f = (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
    z = diff(f, x)
    result = z.subs({x: a, y: b})
    return result


def cal_y_der(a, b):
    x, y = symbols('x y', real=True)
    f = (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
    z = diff(f, y)
    result = z.subs({x: a, y: b})
    return result


def func_arg_min(pre_x, var, error):
    x = pre_x[0] - var * error[0]
    y = pre_x[1] - var * error[1]
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def cal_step_size(pre_x, error, lp, rp, e):
    delta = rp - lp
    gs = (math.sqrt(5)-1)/2
    p1 = rp - gs * delta
    p2 = lp + gs * delta
    fx1 = func_arg_min(pre_x, p1, error)
    fx2 = func_arg_min(pre_x, p2, error)
    k = 0
    while abs(rp-lp) > e:
        if fx1 < fx2:
            rp = p2
            p2 = p1
            fx2 = fx1
            det = rp - p2
            if det >= 1e-4:
                p1 = lp + det
            fx1 = func_arg_min(pre_x, p1, error)
        else:
            lp = p1
            p1 = p2
            fx1 = fx2
            det = p1 - lp
            if det >= 1e-4:
                p2 = rp - det
            fx2 = func_arg_min(pre_x, p2, error)
        k = k + 1
    min_point = (p1+p2)/2
    print("step size", min_point)
    return min_point


# def func(x, y):  # Himmelblau
#     return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
#
#
# def cal_x_der(a, b):
#     x, y = symbols('x y', real=True)
#     f = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
#     z = diff(f, x)
#     result = z.subs({x: a, y: b})
#     return result
#
#
# def cal_y_der(a, b):
#     x, y = symbols('x y', real=True)
#     f = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
#     z = diff(f, y)
#     result = z.subs({x: a, y: b})
#     return result


# def func_arg_min(pre_x, var, temp):
#     x = pre_x[0] + var * temp[0]
#     y = pre_x[1] + var * temp[1]
#     return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


def iter_func(max_iter_count=100000):
    pre_x = [0.0, 0.0]
    pre_xk = [0.0, 0.0]
    error = [0.0, 0.0]

    precision = 1e-4
    iter_count = 0

    # Rosenbrock plt
    a = np.arange(-2, 2, 0.01)
    b = np.arange(-2, 5, 0.01)
    [a, b] = np.meshgrid(a, b)
    f = func(a, b)
    plt.contour(a, b, f, levels=[3, 10, 50, 100, 150], colors='black')
    # # Himmelblau
    # a = np.arange(-5, 5, 0.01)
    # b = np.arange(-5, 5, 0.01)
    # [a, b] = np.meshgrid(a, b)
    # f = func(a, b)
    # plt.contour(a, b, f, levels=[1, 4, 20, 50, 100, 150], colors='black')

    w = np.zeros((10000, 2))
    w[0, :] = pre_x
    while iter_count < max_iter_count:
        f1 = func(pre_x[0], pre_x[1])
        error[0] = round(cal_x_der(pre_x[0], pre_x[1]), 8)
        error[1] = round(cal_y_der(pre_x[0], pre_x[1]), 8)

        step_size = cal_step_size(pre_x, error, 1e-3, 3, 1e-3)

        pre_xk[0] = pre_x[0] - step_size * error[0]
        pre_xk[1] = pre_x[1] - step_size * error[1]

        f2 = func(pre_x[0], pre_x[1])
        if abs(f2-f1) < precision and math.sqrt((pre_xk[0]-pre_x[0])**2+(pre_xk[1]-pre_x[1])**2) < precision and \
                math.sqrt(cal_x_der(pre_xk[0], pre_xk[1])**2+cal_y_der(pre_xk[0], pre_xk[1])**2) < precision:
            # print("iter_count: ", iter_count, " px ", pre_x[0], " py ", pre_x[1])
            break
        else:
            pre_x[0] = round(pre_xk[0], 8)
            pre_x[1] = round(pre_xk[1], 8)
            iter_count += 1
            w[iter_count, :] = pre_x
            print("iter_count: ", iter_count, " px ", pre_x[0], " py ", pre_x[1])
    plt.plot(w[:, 0], w[:, 1], 'r+', w[:, 0], w[:, 1])
    plt.show()
    return pre_x


if __name__ == '__main__':
    iter_func()

