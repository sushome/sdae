# Minimize the function
# f(x, y) = (1-x)^2 + 100 * (y-x^2)^2
# f(x, y) = (x^2+y-11)^2 + (x+y^2-7)^2
import math
import random
import matplotlib.pyplot as plt
import numpy as np


def func(x, y):  # Rosenbrock
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


# def func(x, y):  # Himmelblau
#     return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


def get_dict(res_dict):  # sorted dictionary {key:value}
    # for i in range(3):
    #     x = random.randint(-5, 5)
    #     y = random.randint(-5, 5)
    #     f = func(x, y)
    #     loc = (x, y)
    #     res_dict[loc] = f
    # res_dict = sorted(res_dict.items(), key=lambda item: item[1])
    # Rosenbrock initial data
    res_dict = [((0, -1), 101.0), ((3, -4), 16904.0), ((-5, 3), 48436.0)]
    # Himmelblau initial data
    # res_dict = [((4, -1), 20.0), ((-2, 2), 50.0), ((0, 1), 136.0)]
    # print(type(res_dict))
    # print(res_dict)
    return res_dict


# def sort_dict(res_dict):
#     res_dict = sorted(res_dict.items(), key=lambda item: item[1])
#     # print(res_dict)
#     return res_dict


# def get_max(res_dict):  # get xn+1
#     max_index = len(res_dict)-1
#     return max_index
# def get_second_to_max(res_dict):  # get f(xn)
#     max_second_index = len(res_dict)-2
#     return max_second_index
# def get_min(res_dict):  # get f(x1)
#     min_index = 0
#     return min_index

def get_max_value(res_dict):  # get f(xn+1)
    length = len(res_dict)
    max_value_xy = [res_dict[length-1][0][0], res_dict[length-1][0][1]]
    max_value = func(res_dict[length-1][0][0], res_dict[length-1][0][1])
    # print([max_value_xy, max_value])
    return [max_value_xy, max_value]


def get_second_to_max_value(res_dict):  # get f(xn)
    length = len(res_dict)
    max_second_value_xy = [res_dict[length-2][0][0], res_dict[length-2][0][1]]
    max_second_value = func(res_dict[length-2][0][0], res_dict[length-2][0][1])
    # print([max_second_value_xy, max_second_value])
    return [max_second_value_xy, max_second_value]


def get_min_value(res_dict):  # get f(x1)
    min_value_xy = [res_dict[0][0][0], res_dict[0][0][1]]
    min_value = func(res_dict[0][0][0], res_dict[0][0][1])
    # print([min_value_xy, min_value])
    return [min_value_xy, min_value]


def get_centroid(res_dict):  # get x0
    x = 0
    y = 0
    flag = 0
    length = len(res_dict)
    for res in res_dict:
        flag = flag + 1
        x = x + res[0][0]
        y = y + res[0][1]
        mean_x = x/len(res_dict)
        mean_y = y/len(res_dict)
        mean = [mean_x, mean_y]  # mean list
        while flag == length - 1:
            break
    return mean


def get_reflection(mean, res_dict):
    max_value = get_max_value(res_dict)
    x = max_value[0][0]
    y = max_value[0][1]
    new_cord_x = mean[0]+(mean[0]-x)
    new_cord_y = mean[1]+(mean[1]-y)
    cord_re = [new_cord_x, new_cord_y]
    re = func(new_cord_x, new_cord_y)
    result = [cord_re, re]
    return result


def get_expansion(mean, result_re):
    new_cord_x = mean[0]+2.8*(result_re[0][0]-mean[0])
    new_cord_y = mean[1]+2.8*(result_re[0][1]-mean[1])
    cord_ex = [new_cord_x, new_cord_y]
    ex = func(new_cord_x, new_cord_y)
    result_ex = [cord_ex, ex]
    return result_ex


def get_compression_first(mean, res_dict):
    max_value = get_max_value(res_dict)
    x = max_value[0][0]
    y = max_value[0][1]
    new_cord_x = mean[0]+0.5*(x-mean[0])
    new_cord_y = mean[1]+0.5*(y-mean[1])
    cord_co_f = [new_cord_x, new_cord_y]
    co_f = func(new_cord_x, new_cord_y)
    result_f = [cord_co_f, co_f]
    return result_f


def get_compression_second(mean, result_re):
    new_cord_x = mean[0]+0.5*(result_re[0][0]-mean[0])
    new_cord_y = mean[1]+0.5*(result_re[0][0]-mean[1])
    cord_co_s = [new_cord_x, new_cord_y]
    co_s = func(new_cord_x, new_cord_y)
    result_co_s = [cord_co_s, co_s]
    return result_co_s


def get_shrink(res_dict):
    # n+1
    new_cord_x_n1 = get_min_value(res_dict)[0][0] + \
                 0.5 * (get_max_value(res_dict)[0][0] - get_min_value(res_dict)[0][0])
    new_cord_y_n1 = get_min_value(res_dict)[0][1] + \
                 0.5 * (get_max_value(res_dict)[0][1] - get_min_value(res_dict)[0][1])
    cord_sh_n1 = [new_cord_x_n1, new_cord_y_n1]
    co_sh_n1 = func(new_cord_x_n1, new_cord_y_n1)

    # n
    new_cord_x_n = get_min_value(res_dict)[0][0] + \
                 0.5 * (get_second_to_max_value(res_dict)[0][0]-get_min_value(res_dict)[0][0])
    new_cord_y_n = get_min_value(res_dict)[0][1] + \
                 0.5 * (get_second_to_max_value(res_dict)[0][1]-get_min_value(res_dict)[0][1])
    cord_sh_n = [new_cord_x_n, new_cord_y_n]
    co_sh_n = func(new_cord_x_n, new_cord_y_n)

    # n-1
    new_cord_x = get_min_value(res_dict)[0][0] + \
                 0.5 * (res_dict[1][0][0]-get_min_value(res_dict)[0][0])
    new_cord_y = get_min_value(res_dict)[0][1] + \
                 0.5 * (res_dict[1][0][1]-get_min_value(res_dict)[0][1])
    cord_sh = [new_cord_x, new_cord_y]
    co_sh = func(new_cord_x, new_cord_y)

    result_co_sh = [(cord_sh_n1, co_sh_n1), (cord_sh_n, co_sh_n), (cord_sh, co_sh)]
    return result_co_sh


def delete_supply_dict(x, y, res_dict):
    res_dict.pop()
    temp = {(x, y): func(x, y)}
    res_dict = res_dict + list(temp.items())  # items() from dictionary to list
    res_dict = dict(res_dict)
    res_dict = sorted(res_dict.items(), key=lambda item: item[1])
    print(res_dict)
    return res_dict


if __name__ == '__main__':
    res_dict = {}
    res_dict = get_dict(res_dict)
    e = 10e-5
    while True:
        # max_value = get_max_value(res_dict)
        # min_value = get_min_value(res_dict)
        # distance = math.sqrt(math.pow(max_value[0][0] - min_value[0][0], 2) +
        #                      math.pow(max_value[0][1] - min_value[0][1], 2))
        # if len(res_dict) != 4:
        #     res_dict = {}
        #     res_dict = get_dict(res_dict)
        # distance = math.sqrt((math.pow(res_dict[0][1] - res_dict[3][1], 2) +
        #                      math.pow(res_dict[1][1] - res_dict[3][1], 2) +
        #                      math.pow(res_dict[2][1] - res_dict[3][1], 2)) / 4)
        if len(res_dict) != 3:
            res_dict = {}
            res_dict = get_dict(res_dict)

        distance = math.sqrt((math.pow(res_dict[0][1] - res_dict[2][1], 2) +
                             math.pow(res_dict[1][1] - res_dict[2][1], 2)) / 3)
        if distance <= e:
            x = (res_dict[0][0][0]+res_dict[1][0][0]+res_dict[2][0][0]) / 3
            y = (res_dict[0][0][1] + res_dict[1][0][1] + res_dict[2][0][1]) / 3
            print('convergent point')
            print(x, y)

            # Rosenbrock
            X = np.arange(-3, 3, 0.01)
            Y = np.arange(-2, 6, 0.01)
            [X, Y] = np.meshgrid(X, Y)
            F = func(X, Y)
            plt.contour(X, Y, F, levels=[3, 10, 50, 100, 150], colors='black')

            # Himmelblau
            # X = np.arange(-5, 5, 0.01)
            # Y = np.arange(-5, 5, 0.01)
            # [X, Y] = np.meshgrid(X, Y)
            # F = func(X, Y)
            # plt.contour(X, Y, F, levels=[1, 4, 20, 50, 100, 150], colors='black')

            plt.scatter(x, y, c='red')
            plt.show()
            # print(func(x, y))
            break
        else:
            mean = get_centroid(res_dict)
            result_re = get_reflection(mean, res_dict)
            f_x_k = get_reflection(mean, res_dict)[1]
            f_x_n = get_second_to_max_value(res_dict)[1]
            f_x_n_plus_one = get_max_value(res_dict)[1]
            f_x_1 = get_min_value(res_dict)[1]
            f_x_p = get_expansion(mean, result_re)[1]
            f_x_m_first = get_compression_first(mean, res_dict)[1]
            f_x_m_second = get_compression_second(mean, result_re)[1]
            print(res_dict)
            print(f_x_1)
            print(f_x_k)
            print(f_x_n)
            print('----')
            if f_x_1 <= f_x_k <= f_x_n:
                x = get_reflection(mean, res_dict)[0][0]
                y = get_reflection(mean, res_dict)[0][1]
                res_dict = delete_supply_dict(x, y, res_dict)
                # res_dict.pop()
                # temp = {(x, y): func(x, y)}
                # res_dict = res_dict + list(temp.items())  # items() from dictionary to list
                # res_dict = dict(res_dict)
                # res_dict = sorted(res_dict.items(), key=lambda item: item[1])
                # print(res_dict)
                print('reflection1')
            elif f_x_k < f_x_1:
                if f_x_p < f_x_k:
                    x = get_expansion(mean, result_re)[0][0]
                    y = get_expansion(mean, result_re)[0][1]
                    res_dict = delete_supply_dict(x, y, res_dict)
                    # res_dict.pop()
                    # temp = {(x, y): func(x, y)}
                    # res_dict = res_dict + list(temp.items())
                    # res_dict = dict(res_dict)
                    # res_dict = sorted(res_dict.items(), key=lambda item: item[1])
                    # print(res_dict)
                    print('expansion')
                else:
                    x = get_reflection(mean, res_dict)[0][0]
                    y = get_reflection(mean, res_dict)[0][1]
                    res_dict = delete_supply_dict(x, y, res_dict)
                    # res_dict.pop()
                    # temp = {(x, y): func(x, y)}
                    # res_dict = res_dict + list(temp.items())
                    # res_dict = dict(res_dict)
                    # res_dict = sorted(res_dict.items(), key=lambda item: item[1])
                    # print(res_dict)
                    print('reflection2')
            elif f_x_k > f_x_n:
                if f_x_k >= f_x_n_plus_one:
                    if f_x_m_first < f_x_n_plus_one:
                        x = get_compression_first(mean, res_dict)[0][0]
                        y = get_compression_first(mean, res_dict)[0][1]
                        res_dict = delete_supply_dict(x, y, res_dict)
                        # res_dict.pop()
                        # temp = {(x, y): func(x, y)}
                        # res_dict = res_dict + list(temp.items())
                        # res_dict = dict(res_dict)
                        # res_dict = sorted(res_dict.items(), key=lambda item: item[1])
                        # print(res_dict)
                        print('compression1')
                    else:
                        x1 = get_shrink(res_dict)[0][0][0]
                        y1 = get_shrink(res_dict)[0][0][1]
                        temp1 = {(x1, y1): func(x1, y1)}
                        x2 = get_shrink(res_dict)[1][0][0]
                        y2 = get_shrink(res_dict)[1][0][1]
                        temp2 = {(x1, y1): func(x1, y1)}
                        x3 = get_shrink(res_dict)[2][0][0]
                        y3 = get_shrink(res_dict)[2][0][1]
                        temp3 = {(x1, y1): func(x1, y1)}
                        res_dict.pop()
                        res_dict.pop()
                        res_dict.pop()
                        res_dict = res_dict + list(temp1.items()) + list(temp2.items()) + list(temp3.items())
                        res_dict = dict(res_dict)
                        res_dict = sorted(res_dict.items(), key=lambda item: item[1])
                        print(res_dict)
                        print('new Xn+1')
                else:
                    if f_x_m_second < f_x_k:
                        x = get_compression_second(mean, result_re)[0][0]
                        y = get_compression_second(mean, result_re)[0][1]
                        res_dict = delete_supply_dict(x, y, res_dict)
                        # res_dict.pop()
                        # temp = {(x, y): func(x, y)}
                        # res_dict = res_dict + list(temp.items())
                        # res_dict = dict(res_dict)
                        # res_dict = sorted(res_dict.items(), key=lambda item: item[1])
                        # print(res_dict)
                        print('compression2')
                    else:
                        x = get_reflection(mean, res_dict)[0][0]
                        y = get_reflection(mean, res_dict)[0][1]
                        res_dict = delete_supply_dict(x, y, res_dict)
                        # res_dict.pop()
                        # temp = {(x, y): func(x, y)}
                        # res_dict = res_dict + list(temp.items())
                        # res_dict = dict(res_dict)
                        # res_dict = sorted(res_dict.items(), key=lambda item: item[1])
                        # print(res_dict)
                        print('reflection4')
