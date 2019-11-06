# Minimize the function
# f(x, y) = (1-x)^2 + 100 * (y-x^2)^2
# f(x, y) = (x^2+y-11)^2 + (x+y^2-7)^2
import math
import numpy as np
import matplotlib.pyplot as plt
from sympy import diff, symbols, solve


# def func(x, y):  # Rosenbrock
#     return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
#
#
# def cal_x_der(a, b):
#     x, y = symbols('x y', real=True)
#     f = (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
#     z = diff(f, x)
#     result = z.subs({x: a, y: b})
#     return result
#
#
# def cal_y_der(a, b):
#     x, y = symbols('x y', real=True)
#     f = (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
#     z = diff(f, y)
#     result = z.subs({x: a, y: b})
#     return result
#
#
# def cal_step_size(pre_x, error):
#     h = symbols('h', real=True)
#     f = (1-(pre_x[0]-h*error[0])) ** 2 + 100 * (pre_x[1]-h*error[1] - (pre_x[0]-h*error[0]) ** 2) ** 2
#     z = diff(f, h)   # the first derivative of the function
#     s = solve(z, h)  # get the root of the equation
#     c = 0
#     for i in range(len(s)):
#         if complex(s[i]).imag == 0:
#             s = float(s[i])
#             c = c + 1
#     if c == 0:
#         s = 0.001
#     return s


def func(x, y):  # Himmelblau
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


def cal_x_der(a, b):
    x, y = symbols('x y', real=True)
    f = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
    z = diff(f, x)
    result = z.subs({x: a, y: b})
    return result


def cal_y_der(a, b):
    x, y = symbols('x y', real=True)
    f = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
    z = diff(f, y)
    result = z.subs({x: a, y: b})
    return result


def cal_step_size(pre_x, error):
    h = symbols('h', real=True)
    f = ((pre_x[0]-h*error[0]) ** 2 + pre_x[1]-h*error[1] - 11) ** 2 + \
        (pre_x[0]-h*error[0] + (pre_x[1]-h*error[1]) ** 2 - 7) ** 2
    z = diff(f, h)   # the first derivative of the function
    s = solve(z, h)  # get the root of the equation
    c = 0
    for i in range(len(s)):
        if complex(s[i]).imag == 0:
            s = float(s[i])
            c = c + 1
    if c == 0:
        s = 0.01
    return s


def iter_func(max_iter_count=100000):
    pre_x = [0.0, 0.0]
    pre_xk = [0.0, 0.0]
    error1 = [0.0, 0.0]
    error2 = [0.0, 0.0]
    precision = 1e-4
    iter_count = 0
    # # Rosenbrock plt
    # a = np.arange(-2, 2, 0.01)
    # b = np.arange(-2, 5, 0.01)
    # [a, b] = np.meshgrid(a, b)
    # f = func(a, b)
    # plt.contour(a, b, f, levels=[3, 10, 50, 100, 150], colors='black')
    # Himmelblau
    a = np.arange(-5, 5, 0.01)
    b = np.arange(-5, 5, 0.01)
    [a, b] = np.meshgrid(a, b)
    f = func(a, b)
    plt.contour(a, b, f, levels=[1, 4, 20, 50, 100, 150], colors='black')

    w = np.zeros((100, 2))
    w[0, :] = pre_x

    error1[0] = round(cal_x_der(pre_x[0], pre_x[1]), 8)
    error1[1] = round(cal_y_der(pre_x[0], pre_x[1]), 8)
    while iter_count < max_iter_count:
        # if error1[0] == 0.0 and error1[1] == 0.0:
        #     print("iter_count: ", iter_count, " px ", pre_x[0], " py ", pre_x[1])
        #     break
        f1 = func(pre_x[0], pre_x[1])
        step_size = cal_step_size(pre_x, error1)

        pre_xk[0] = pre_x[0] - step_size * error1[0]
        pre_xk[1] = pre_x[1] - step_size * error1[1]

        error2[0] = round(cal_x_der(pre_xk[0], pre_xk[1]), 8)
        error2[1] = round(cal_y_der(pre_xk[0], pre_xk[1]), 8)
        b = math.sqrt(error2[0]**2 + error2[1]**2)/math.sqrt(error1[0]**2 + error1[1]**2)

        error1[0] = error2[0] + b * error1[0]
        error1[1] = error2[1] + b * error1[1]

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

