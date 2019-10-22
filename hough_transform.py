import numpy as np
import math
import matplotlib.pyplot as plt


def create_data(flag=0):
    if flag == 0:
        x = np.arange(-10, 10, 1)
        randnum = np.random.randint(0, 2, np.size(x))
        y = -x + np.random.random(np.size(x)) * randnum
        return x, y
    if flag == 1:
        x = np.arange(1, 11, 1)

        y = np.array([1, 0, 5, 4, 4.5, 6, 9, 8, 9, 7])
        return x, y


def hough_line(points, angle_step=1):
    # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(0.0, 180.0, angle_step))

    (x, y) = points
    width = np.max(x) - np.min(x)
    height = np.max(y) - np.min(y)

    diag_len = int(round(math.sqrt(width * width + height * height)))+1
    # 这里选择使用了2*diag_len的行来存储
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)

    # Cache some resuable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len , num_thetas), dtype=np.uint8)

    # Vote in the hough accumulator
    for i in range(len(x)):
        x_value = x[i]
        y_value = y[i]
        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            # rho = xcost+ysint
            rho = diag_len + int(round(x_value * cos_t[t_idx] + y_value * sin_t[t_idx]))
            accumulator[rho, t_idx] += 1

    return accumulator, thetas, rhos,diag_len


def show_hough_line(points, accumulator, thetas, rhos, diag_len,save_path=None):
    plt.scatter(points[0], points[1])
    index = np.argmax(accumulator)
    index_x = int(index / np.size(accumulator, 1))
    index_y = index - index_x * np.size(accumulator, 1)
    rho, theta = index_x-diag_len, thetas[index_y]

    k = -np.cos(theta) / np.sin(theta)
    b = rho / np.sin(theta)
    plt.plot(points[0], k * points[0] + b)

    if b>0:
        print("y= %f*x+ %f " % (k, b))
    else:
        print("y= %f*x %f " % (k, b))
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    points = create_data()

    accumulator, thetas, rhos, diag_len = hough_line(points)
    show_hough_line(points, accumulator, thetas, rhos, diag_len,save_path='output/res.png')
