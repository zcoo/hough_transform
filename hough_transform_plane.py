import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def create_data():
    # 生成一个z=1的平面
    x = 10 * np.random.rand(100)
    y = 10 * np.random.rand(100)
    z = -0.5*x + 0.001 * np.random.rand(100)
    # 加一点点噪声
    z[0], z[1], z[2], z[3], z[4] = 1.1, -0.1, 1.7, 1.3, 0.7
    return x, y, z


def hough_plane(points, angle_step=1):
    # Rho, phi and Theta ranges

    thetas = np.deg2rad(np.arange(0.0, 180.0, angle_step))
    phis = np.deg2rad(np.arange(-180.0, 180.0, angle_step))

    (x, y, z) = points
    width = np.max(x) - np.min(x)
    length = np.max(y) - np.min(y)
    height = np.max(z) - np.min(z)

    diag_len = int(round(math.sqrt(
        width * width + length * length + height * height))) + 1

    # Cache some resuable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    cos_p = np.cos(phis)
    sin_p = np.sin(phis)
    num_thetas = len(thetas)
    num_phis = len(phis)

    # Hough accumulator array of theta, phis vs rho
    accumulator = np.zeros((3 * diag_len, num_thetas, num_phis), dtype=np.uint8)

    # Vote in the hough accumulator
    # for i in range(len(x)):
    #     x_value = x[i]
    #     y_value = y[i]
    #     z_value = z[i]
    #     for t_idx1 in range(num_thetas):
    #         for t_idx2 in range(num_phis):
    #             # Calculate rho. diag_len is added for a positive index
    #             # rho = xsintcosp+ysintsinp+zcost
    #             rho = diag_len + int(round(x_value * sin_t[t_idx1] * cos_p[t_idx2]
    #                                        + y_value * sin_t[t_idx1] * sin_p[t_idx2] + z_value * cos_t[t_idx1]))
    #             accumulator[rho, t_idx1, t_idx2] += 1

    rho = diag_len + np.round(x[:,None,None] * sin_t[:, None]* cos_p[None, :]
                              + y[:,None,None] * sin_t[:, None]* sin_p[None, :] + z[:,None,None] * cos_t[:, None])
    rho = rho.astype(int)
    for i in range(len(x)):
        for t_idx1 in range(num_thetas):
            for t_idx2 in range(num_phis):
                accumulator[rho[i,t_idx1,t_idx2], t_idx1, t_idx2] += 1


    return accumulator, thetas, phis, diag_len


def visulize(points, accumulator, thetas, phis, diag_len, save_path=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points[0], points[1], points[2], c='r', marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    X, Y = np.meshgrid(points[0], points[1])

    index = np.unravel_index(accumulator.argmax(), accumulator.shape)
    rho, theta, phi = index[0] - diag_len, thetas[index[1]], phis[index[2]]

    a = np.sin(theta) * np.cos(phi)
    b = np.sin(theta) * np.sin(phi)
    c = np.cos(theta)
    d = -rho

    Z = -a / c * X - b / c * Y - d / c

    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    print("%f *x + %f *y + %f *z +%f = 0" % (a, b, c, d))

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':

    points = create_data()

    accumulator, thetas, phis, diag_len = hough_plane(points)
    visulize(points, accumulator, thetas, phis, diag_len, save_path='output/res.png')
