import numpy as np
import matplotlib.pyplot as plt
def XYFunction(x0, y0, T, rate, eta_x, epsilon, n, eta, gamma):
    """

    :param T: 总时长
    :param rate: 每次增加的时间
    :return:
    """
    X = [x0]
    Y = [y0]
    total_t = 0
    for i in range(int(T/rate)):
        x_temp = (eta_x - X[-1])*rate + X[-1]
        y_temp = (epsilon + eta/(1 + pow(X[-1] , n) * max((total_t - gamma), 0)) - Y[-1]) * rate + Y[-1]
        X.append(x_temp)
        Y.append(y_temp)
        total_t = total_t + rate
    return X, Y


if __name__ == "__main__":
    T = 10
    x0, y0 = 0, 6
    rate = 0.1
    eta_x = 6
    epsilon, n, eta, gamma = 0.5, 2, 5.5, 2
    X, Y = XYFunction(x0, y0, T, rate, eta_x, epsilon, n, eta, gamma)
    plt_t = np.arange(-1*rate, T, rate)
    plt.figure()
    plt.plot(plt_t, X, label="X")
    plt.plot(plt_t, Y, label="Y", linestyle="--")
    # plt.plot(x, obs_list[6], label="x7", linestyle="--")
    # plt.plot(x, obs_list[3], label="x3", linestyle="-")
    # plt.plot(x, goal_list[0].squeeze(), label="x3-goal", linestyle="-")
    # plt.plot(x, obs_list2[1], label="x12", linestyle=":")
    # plt.plot(x, action[0], label="action", linestyle=":")
    plt.legend(loc='upper left')
    plt.savefig('./test2.jpg')



