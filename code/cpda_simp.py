import numpy as np
import math
from scipy.signal import convolve2d
from scipy.ndimage import label


def cpda_simp(BW):
    I = BW
    EP = 0

    # Extract curves from the edge-image
    curve, curve_start, curve_end, curve_mode, curve_num, TJ, img1 = extract_curve(BW)

    sizex, sizey = I.shape
    if curve and len(curve[0]) > 0:
        # Detect corners on the extracted edges
        corner_out, index, Sig, cd2 = get_corner(curve, curve_mode, curve_start, curve_num, sizex, sizey)

        # Update the T-junctions
        corner_final, cd3 = refine_t_junctions(corner_out, TJ, cd2, curve, curve_num, curve_start, curve_end, curve_mode, EP)

        cout = corner_final
        cd = cd3
    else:
        cout = np.array([])
        marked_img = []
        cd = []

    here = 1

    return cout


def get_corner(curve, curve_mode, curve_start, curve_num, sizex, sizey):
    corners = np.array([])
    cor = np.array([])  # candidate corners
    cd = np.array([])
    T_angle = 157
    CLen = [10, 20, 30]
    T = 0.2  # define the curvature threshold
    sig = 3.0
    gau, W = makeGFilter(sig)
    index = {}
    Sig = np.zeros((curve_num, 1))

    for i in range(curve_num):
        C = np.array([])
        x = curve[i][:, 1] - sizey / 2
        y = sizex / 2 - curve[i][:, 0]
        curveLen = len(x)
        # smooth the curve with Gaussian kernel
        xs, ys, W = smoothing(x, y, curveLen, curve_mode[i], gau, W)

        if len(xs) > 1:
            if curve_mode[i] == 'loop':
                xs1 = np.concatenate([xs[curveLen - W:curveLen], xs, xs[0:W]])
                ys1 = np.concatenate([ys[curveLen - W:curveLen], ys, ys[0:W]])
            else:
                first_x1 = np.ones((W, 1)) * 2 * xs[0]
                second_x1 = np.ones((W, 1)) * 2 * xs[-1]
                first_y1 = np.ones((W, 1)) * 2 * ys[0]
                second_y1 = np.ones((W, 1)) * 2 * ys[-1]
                for k in range(W):
                    first_x1[k] -= xs[W - k]
                    second_x1[k] -= xs[- 2 - k]
                    first_y1[k] -= ys[W - k]
                    second_y1[k] -= ys[- 2 - k]
                xs1 = np.concatenate([first_x1.ravel(), xs, second_x1.ravel()])
                ys1 = np.concatenate([first_y1.ravel(), ys, second_y1.ravel()])
            xs = xs1
            ys = ys1
            L = curveLen + 2 * W
            C3 = np.zeros((3, L))

            for j in range(3):
                chordLen = CLen[j]
                C3[j, 0:L] = np.abs(accumulate_chord_distance(xs, ys, chordLen, L))

            c1 = C3[0, W:curveLen + W] / np.max(C3[0, W:curveLen + W])
            c2 = C3[1, W:curveLen + W] / np.max(C3[1, W:curveLen + W])
            c3 = C3[2, W:curveLen + W] / np.max(C3[2, W:curveLen + W])

            C = c1 * c2 * c3
            L = curveLen
            xs = xs[W:L + W]
            ys = ys[W:L + W]

            N = len(C)
            extremum = np.zeros(N)
            n = 0
            Search = 1
            smoothed_curve = np.vstack((xs, ys)).T

            for j in range(N - 1):
                if (C[j + 1] - C[j]) * Search > 0:
                    extremum[n] = j      # In extremum, odd points are minima and even points are maxima
                    Search = -Search        # minima: when K starts to go up; maxima: when K starts to go down
                    n += 1
            index_2 = 0
            for k in range(1, N - 1):
                if(extremum[k] == 0):
                    index_2 = k
                    break
            extremum = extremum[0:index_2].astype(int)

            if extremum.size % 2 != 0:
                extremum[n] = N
                n += 1
            cols = extremum.size
            n = cols
            flag = np.ones(cols)

            for j in range(1, n, 2):
                if C[extremum[j]] > T:
                    flag[j] = 0

            extremum = extremum[1:n:2]
            flag = flag[1:n:2]
            extremum = extremum[flag == 0]

            for j in range(extremum.size):
                if j == 0:
                    cor = curve[i][extremum[j], :]
                else:
                    cor = np .vstack([cor, curve[i][extremum[j], :]])
            flag = np.zeros(cols)

            while np.sum(flag==0)>0:
                n = extremum.shape[0]
                flag = np.ones(n)
                for j in range(n):
                    if j == 0 and j == n - 1:
                        ang = curve_tangent(smoothed_curve[0:L, :], extremum[j])
                    elif j == 0:
                        ang = curve_tangent(smoothed_curve[0:extremum[j + 1]], extremum[j])
                    elif j == n - 1:
                        ang = curve_tangent(smoothed_curve[extremum[j - 1]-1:L], extremum[j] - extremum[j - 1] + 1)
                    else:
                        ang = curve_tangent(smoothed_curve[extremum[j - 1] - 1:extremum[j + 1]],
                                            extremum[j] - extremum[j - 1] + 1)

                    if T_angle < ang < (360 - T_angle):
                        flag[j] = 0

                if len(extremum.shape) == 0:
                    extremum.clear()
                else:
                    extremum = extremum[flag != 0]
            extremum = [e for e in extremum if 0 < e <= curveLen]

            index[i] = np.transpose(extremum)
            Sig[i, 0] = sig

            for j in range(n):
                if j == 0:
                    corners = curve[i][extremum[j]]
                    cd = C[extremum[j]]
                else:
                    corners = np.vstack((corners, curve[i][extremum[j]]))
                    cd = np.vstack((cd, C[extremum[j]]))

            if curve_mode[i] == 'loop':
                if n > 1:
                    compare_corner = corners - np.ones((corners.shape[0], 1)) * curve_start[i]
                    compare_corner = compare_corner ** 2
                    compare_corner = compare_corner[:, 0] + compare_corner[:, 1]
                    if np.min(compare_corner) > 100:
                        left = smoothed_curve[extremum[0]:0:-1, :]
                        right = smoothed_curve[-1:extremum[-1]-1:-1, :]
                        ang = curve_tangent(np.vstack(left, right), extremum[0])
                        if T_angle < ang < (360 - T_angle):
                            pass
                        else:
                            corners = np.vstack(corners, curve_start[i])
                            cd = np.vstack(cd, 5)

    return corners, index, Sig, cd


# 累积每个点处指定长度的弦到该点的距离，生成一个描述曲线特征的向量。
def accumulate_chord_distance(xs, ys, chord_len, curve_len):
    Cd = np.zeros(curve_len)

    for k in range(1, curve_len):
        xk = xs[k]  # (x1, y1) = point at which distance will be accumulated
        yk = ys[k]

        s = max(1, k - chord_len + 1)
        for i in range(s-1, k):
            if i + chord_len < curve_len:
                x1 = xs[i]  # (leftx, lefty) = current left point for which distance will be accumulated
                y1 = ys[i]

                x2 = xs[i + chord_len]      # (rightx, righty) = current right point for which distance will be accumulated
                y2 = ys[i + chord_len]

                a = y2 - y1  # coefficients of straight line through points (x1, y1) and (x2, y2)
                b = x1 - x2
                c = x2 * y1 - x1 * y2
                dist = (a * xk + b * yk + c) / np.sqrt(a * a + b * b)
                Cd[k] += dist
            else:
                break

    return Cd


# 将给定的曲线（由输入参数 xs 和 ys 表示）在两端进行放大
def enlarge(xs, ys, CL, curve_mode):
    L = len(xs)

    if curve_mode == 'loop':
        xse = np.concatenate([xs[L-CL:L], xs, xs[0:CL]])
        yse = np.concatenate([ys[L-CL:L], ys, ys[0:CL]])
    else:
        xse = np.concatenate([np.ones(CL, 1) * (2 * xs[0]) - xs[CL+1:1:-1], xs, np.ones(CL) * (2 * xs[L-1]) - xs[L-1:L-CL-1:-1]])
        yse = np.concatenate([np.ones(CL, 1) * (2 * ys[0]) - ys[CL+1:1:-1], ys, np.ones(CL) * (2 * ys[L-1]) - ys[L-1:L-CL-1:-1]])

    return xse, yse


# 平滑处理
def smoothing(x, y, L, curve_mode, gau, W):
    if L > W:
        # wrap around the curve by W pixles at both ends
        if curve_mode == 'loop':
            x1 = np.concatenate((x[L-W:L], x, x[0:W]))
            y1 = np.concatenate((y[L-W:L], y, y[0:W]))
        # extend each line curve by W pixels at both ends
        else:
            first_x1 = np.ones((W, 1)) * 2 * x[0]
            second_x1 = np.ones((W, 1)) * 2 * x[L - 1]
            first_y1 = np.ones((W, 1)) * 2 * y[0]
            second_y1 = np.ones((W, 1)) * 2 * y[L - 1]
            for i in range(W):
                first_x1[i] -= x[W - i]
                second_x1[i] -= x[L - 2 - i]
                first_y1[i] -= y[W - i]
                second_y1[i] -= y[L - 2 - i]
            x1 = np.concatenate([first_x1.ravel(), x, second_x1.ravel()])
            y1 = np.concatenate([first_y1.ravel(), y, second_y1.ravel()])

        xx = np.convolve(x1, gau, mode='full')
        xs = xx[2*W:L+W*2]
        yy = np.convolve(y1, gau, mode='full')
        ys = yy[2*W:L+W*2]
    else:
        xs = []
        ys = []

    return xs, ys, W


# 模拟matlab的find函数，先列后行
def find(BW, x_min, x_max, y_min , y_max, k):
    cols = np.array([])
    rows = np.array([])
    for i in range(y_min, y_max):
        for j in range(x_min, x_max):
            if BW[j, i] == k:
                cols = np.append(cols, i - y_min)
                rows = np.append(rows, j - x_min)

    return rows, cols


# extract curves from input edge-image
def extract_curve(BW):
    L, W = BW.shape
    BW1 = np.zeros((L + 2, W + 2))
    BW_edge = np.zeros((L, W))
    BW1[1:L + 1, 1:W + 1] = BW
    r, c = np.where(BW1 == 1)
    cur_num = -1
    curve = {}

    while r.size > 0:
        point = np.array([r[0], c[0]])
        cur = np.array([point])
        BW1[point[0], point[1]] = 0

        I, J = find(BW1, point[0] - 1, point[0] + 2, point[1] - 1, point[1] + 2, 1)

        while I.size > 0:
            dist = (I - 1) ** 2 + (J - 1) ** 2
            index = np.argmin(dist)
            point = point + np.array([I[index], J[index]]) - 1  # next is the current point
            cur = np.vstack([cur, point])
            point = point.astype(int)
            BW1[point[0], point[1]] = 0
            I, J = find(BW1, point[0] - 1, point[0] + 2, point[1] - 1, point[1] + 2, 1)  # 遍历周围的像素点以查询边界

        # Extract edge towards another direction
        point = np.array([r[0], c[0]])
        BW1[point[0], point[1]] = 0
        I, J = find(BW1, point[0] - 1, point[0] + 2, point[1] - 1, point[1] + 2, 1)


        while I.size > 0:
            dist = (I - 1) ** 2 + (J - 1) ** 2
            index = np.argmin(dist)
            point = point + np.array([I[index], J[index]]) - 1
            cur = np.vstack([cur, point])
            point = point.astype(int)
            BW1[point[0], point[1]] = 0
            I, J = np.where(BW1[point[0] - 1:point[0] + 2, point[1] - 1:point[1] + 2] == 1)

        if cur.shape[0] > (L + W) / 25:  # for a 512 by 512 image, choose curve if its length > 40
            cur_num += 1
            curve[cur_num] = np.array(cur) - 1

        r, c = np.where(BW1 == 1)

    curve_start = np.zeros((cur_num + 1, 2), dtype=int)
    curve_end = np.zeros((cur_num + 1, 2), dtype=int)
    curve_mode = np.empty(cur_num + 1, dtype=f'U{4}')
    BW_edge = np.zeros_like(BW, dtype=int)

    for i in range(cur_num + 1):
        current_curve = curve[i]
        curve_end[i, :] = current_curve[0, :]
        curve_start[i, :] = current_curve[-1, :]

        sum = np.sum((curve_start[i] - curve_end[i]) ** 2)
        if i <= cur_num - 1 and sum <= 25:
            curve_mode[:] = 'loop'
        else:
            curve_mode[:] = 'line'

        index_1 = current_curve[:, 0] * W + (current_curve[:, 1])
        for i in range(len(index_1)):
            BW_edge[(index_1[i] // W).astype(int), (index_1[i] % W).astype(int)] = 1

    cur_num += 1
    TJ = None
    if cur_num == 0:
        curve = np.array([])
        curve_start = np.array([])
        curve_end = np.array([])
        curve_mode = np.array([])
        cur_num = np.array([])
        TJ = None

    img = np.where(BW_edge == 0, 1, 0)

    return curve, curve_start, curve_end, curve_mode, cur_num, TJ, img


# 将输入的T形交叉点 TJ 添加到角点集 corner_out 中，并更新与角点相关的参数 c2 和 c3。
def refine_t_junctions(corner_out, TJ, c2, curve, curve_num, curve_start, curve_end, curve_mode, EP):
    c3 = c2.copy()

    corner_final = corner_out.copy()
    if corner_final is not None and corner_final.any():
        rows = corner_final.shape
    # Add T-junctions
    if TJ != None:
        for i in range(TJ.shape[0]):
            # T-junctions compared with detected corners
            if rows > 0:
                compare_corner = corner_final - np.ones((corner_final.shape[0], 1)) * TJ[i, :]
                compare_corner = compare_corner**2
                compare_corner = np.sum(compare_corner[:, :2], axis=1)
                if np.min(compare_corner) > 100:  # Add end points far from detected corners, i.e. outside of 5 by 5 neighbor
                    corner_final = np.vstack((corner_final, TJ[i, :]))
                    c3 = np.vstack((c3, 10))
            else:
                corner_final = np.vstack((corner_final, TJ[i, :]))
                c3 = np.vstack((c3, 10))

    return corner_final, c3


# show corners into the output images or into the edge-image
# 标记图像
def mark(img, x, y, w):
    M, N, C = img.shape
    img1 =img
    x_start = max(1, x - w // 2)
    x_end = min(M, x + w // 2)
    y_start = max(1, y - w // 2)
    y_end = min(N, y + w // 2)
    if isinstance(img, bool):
        img1[x_start - 1:x_end, y_start - 1:y_end, :] = \
            (img1[x_start - 1:x_end, y_start - 1:y_end, :] < 1)
        img1[x - w // 2:x + w // 2, y - w // 2:y + w // 2, :] = \
            img[x - w // 2:x + w // 2, y - w // 2:y + w // 2, :]
    else:
        img1[x_start - 1:x_end, y_start - 1:y_end, :] = \
            (img1[x_start - 1:x_end, y_start - 1:y_end, :] < 128) * 255
        img1[x - w // 2:x + w // 2, y - w // 2:y + w // 2, :] = \
            img[x - w // 2:x + w // 2, y - w // 2:y + w // 2, :]

    return img1


# 计算两条切线之间的夹角
def curve_tangent(cur, center):
    direction = np.zeros(2)
    for i in range(0, 2):
        if i == 0:
            curve = cur[center::-1, :]
        else:
            curve = cur[center:, :]

        L = curve.shape[0]

        if L > 3:
            # if not collinear
            if(np.sum(curve[0, :] != curve [L - 1, :])) != 0:
                M = int(np.ceil(L / 2))
                x1 = curve[0, 0]
                y1 = curve[0, 1]
                x2 = curve[M-1, 0]
                y2 = curve[M-1, 1]
                x3 = curve[L - 1, 0]
                y3 = curve[L - 1, 1]
            else:
                M1 = int(np.ceil(L / 3))
                M2 = int(np.ceil(2 * L / 3))
                x1 = curve[0, 0]
                y1 = curve[0, 1]
                x2 = curve[M1 - 1, 0]
                y2 = curve[M1 - 1, 1]
                x3 = curve[M2 - 1, 0]
                y3 = curve[M2 - 1, 1]
            if abs((x1 - x2) * (y1 - y3) - (x1 - x3) * (y1 - y2)) < 1e-8:  # straight line
                tangent_direction = np.angle(complex(curve[L - 1, 0] - curve[0, 0], curve[L - 1, 1] - curve[0, 1])) # 计算复数弧度
            else:
                # Fit a circle
                x0 = 1 / 2 * (
                        -y1 * x2 ** 2 + y3 * x2 ** 2 - y3 * y1 ** 2 - y3 * x1 ** 2 - y2 * y3 ** 2 + x3 ** 2 * y1 +
                        y2 * y1 ** 2 - y2 * x3 ** 2 - y2 ** 2 * y1 + y2 * x1 ** 2 + y3 ** 2 * y1 + y2 ** 2 * y3) / (
                             -y1 * x2 + y1 * x3 + y3 * x2 + x1 * y2 - x1 * y3 - x3 * y2)
                y0 = -1 / 2 * (
                        x1 ** 2 * x2 - x1 ** 2 * x3 + y1 ** 2 * x2 - y1 ** 2 * x3 + x1 * x3 ** 2 - x1 * x2 ** 2 -
                        x3 ** 2 * x2 - y3 ** 2 * x2 + x3 * y2 ** 2 + x1 * y3 ** 2 - x1 * y2 ** 2 + x3 * x2 ** 2) / (
                             -y1 * x2 + y1 * x3 + y3 * x2 + x1 * y2 - x1 * y3 - x3 * y2)
                # R = np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)

                radius_direction = np.angle(complex(x0 - x1, y0 - y1))
                if radius_direction < 0:
                    radius_direction = 2 * np.pi - abs(radius_direction)

                adjacent_direction = np.angle(complex(x2 - x1, y2 - y1))

                if adjacent_direction < 0:
                    adjacent_direction = 2 * np.pi - abs(adjacent_direction)

                tangent_direction = np.sign(
                    np.sin(adjacent_direction - radius_direction)) * np.pi / 2 + radius_direction
                if tangent_direction < 0:
                    tangent_direction = 2 * np.pi - abs(tangent_direction)
                elif tangent_direction > 2 * np.pi:
                    tangent_direction = tangent_direction - 2 * np.pi

        else:  # very short line
            tangent_direction = np.angle(complex(curve[L - 1, 0] - curve[0, 0], curve[L - 1, 1] - curve[0, 1]))


        direction[i] = tangent_direction * 180 / np.pi

    ang = abs(direction[0] - direction[1])
    return ang


# 生成高斯滤波器
def makeGFilter(sig):
    GaussianDieOff = 0.00005
    pw = np.arange(1, 101)

    ssq = sig * sig
    W = np.argmax(np.where(np.exp(-(pw ** 2) / (2 * ssq)) > GaussianDieOff))
    if W == 0:
        W = 1
    t = np.arange(-W, W+1)
    gau = np.exp(-(t ** 2) / (2 * ssq)) / (2 * math.pi * ssq)
    G = gau / sum(gau)
    return G, W


