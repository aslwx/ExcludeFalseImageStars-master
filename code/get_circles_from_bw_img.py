import numpy as np
from scipy.linalg import solve
from scipy.ndimage import label
from scipy.optimize import least_squares


def get_circles_from_bw_img(bw_img):

    # 从阈值图像中获取圆
    # 输入阈值图像，输出圆
    L , W = bw_img.shape
    labeled_img, num_objects = label(bw_img, structure=np.ones((3, 3), dtype=bool))  # 连通集，即边缘
    cc_cell = [np.column_stack(np.where(labeled_img == i + 1)) for i in range(num_objects)]  # 边缘具体位置，列表中每个元素为每个天体边缘的位置
    circles = []
    matchEdge = np.zeros(((num_objects * num_objects - num_objects) // 2, 3))
    count = 0
    cc_cell_adjust = {}
    for i in range(num_objects):
        cc_cell_adjust[i] = [np.sort(cc_cell[i][:, 1] * L + cc_cell[i][:, 0] + 1)]

    # 判定同一天体的组合
    for j in range(num_objects - 1):
        for k in range(j + 1, num_objects):

            # 计算每个线性索引对应的行列坐标
            x1 = np.floor((cc_cell_adjust[j][0] - 1) / 1024)
            y1 = np.mod((cc_cell_adjust[j][0] - 1), 1024)
            x2 = np.floor((cc_cell_adjust[k][0] - 1) / 1024)
            y2 = np.mod((cc_cell_adjust[k][0] - 1), 1024)

            x = np.concatenate([x1, x2])
            y = np.concatenate([y1, y2])

            N = len(x)
            xx = x * x
            yy = y * y
            xy = x * y

            A = np.array([
                [np.sum(x), np.sum(y), N],
                [np.sum(xy), np.sum(yy), np.sum(y)],
                [np.sum(xx), np.sum(xy), np.sum(x)]
            ])

            B = np.array([
                -np.sum(xx + yy),
                -np.sum(xx * y + yy * y),
                -np.sum(xx * x + xy * y)
            ])

            a = solve(A, B)

            xC = -0.5 * a[0]
            yC = -0.5 * a[1]
            R = np.sqrt(-(a[2] - xC ** 2 - yC ** 2))

            # 计算每个点的误差值
            Diff = np.zeros((len(x), 1))
            for l in range(len(x)):
                Diff[l] = np.linalg.norm([x[l] - xC, y[l] - yC]) - R

            # 使用残差最大值
            Diff = np.abs(Diff)
            indices = np.argsort(Diff)[::-1]
            Diff = Diff[indices]
            bwRange = np.mean(Diff[:5]) / R  # 误差最大的5项的平均值除以半径

            # 判定,统一天体标记为1
            if bwRange < 0.0923:
                bwRange = 1
            else:
                bwRange = 0

            matchEdge[count, 0] = j
            matchEdge[count, 1] = k
            matchEdge[count, 2] = bwRange
            count = count + 1

    # 组合标记为同一天体的边缘的连通集
    for j in range(num_objects - 1):
        for k in range(j + 1, num_objects):
            if np.any((matchEdge[:, 0] == j) & (matchEdge[:, 1] == k) & (matchEdge[:, 2] == 1)):
                cc_cell_adjust[k][0] = np.concatenate((cc_cell_adjust[k][0], cc_cell_adjust[j][0]))
                cc_cell_adjust[j][0] = np.array([])
                break

    count = 0
    for i in range(num_objects):
        if cc_cell_adjust[i][0].shape[0] != 0:
            count += 1

    # 初始化 circles 数组
    circles = np.zeros((count, 3))

    # 重新设置 count 为 0
    count = 0

    # 组合后的边缘重新拟合
    for j in range(num_objects):
        if cc_cell_adjust[j][0].shape[0] != 0:
            x = np.floor((cc_cell_adjust[j][0] - 1) / 1024) + 1
            y = np.mod((cc_cell_adjust[j][0] - 1), 1024) + 1
            N = len(x)
            xx = x * x
            yy = y * y
            xy = x * y
            A = np.array([
                [np.sum(x), np.sum(y), N],
                [np.sum(xy), np.sum(yy), np.sum(y)],
                [np.sum(xx), np.sum(xy), np.sum(x)]
            ])
            B = np.array([-np.sum(xx + yy), -np.sum(xx * y + yy * y), -np.sum(xx * x + xy * y)])
            a = np.linalg.solve(A, B)
            xC = -0.5 * a[0]
            yC = -0.5 * a[1]
            R = np.sqrt(-(a[2] - xC ** 2 - yC ** 2))

            circles[count, :] = [xC, yC, R]
            count += 1

    return np.array(circles)
