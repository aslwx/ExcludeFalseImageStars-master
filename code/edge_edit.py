import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage.measure import regionprops, label
from skimage.morphology import binary_erosion, binary_dilation
from scipy.ndimage import convolve
from scipy import signal
from scipy.ndimage import label, generate_binary_structure


def edge_edit(med_img, str):
    # Parse inputs
    a, method, thresh, sigma, thinning, H, kx, ky = parse_inputs(med_img, str)

    # Check valid number of output arguments
    if method not in ['sobel', 'roberts', 'prewitt']:
        raise ValueError("Too many output arguments")

    # Transform to a double precision intensity image if necessary
    if a.dtype != np.float64 and a.dtype != np.float32:
        a = a.astype(np.float32)

    m, n = a.shape

    if method == 'roberts':
        x_mask = np.array([[1, 0], [0, -1]]) / 2  # Roberts approximation to diagonal derivative
        y_mask = np.array([[0, 1], [-1, 0]]) / 2

        scale = 6
        offset = np.array([-1, 1, 1, -1])
    else:
        raise ValueError("Invalid edge detection method")

    # Compute the gradient in x and y direction
    bx = -convolve(a, x_mask, mode='constant', cval=0.0)
    by = -convolve(a, y_mask, mode='constant', cval=0.0)

    # Compute the magnitude
    b = kx * bx ** 2 + ky * by ** 2

    # Determine the threshold; see page 514 of "Digital Imaging Processing" by
    # William K. Pratt
    if len(thresh) == 0:  # Determine cutoff based on RMS estimate of noise
        # Mean of the magnitude squared image is a
        # value that's roughly proportional to SNR
        cutoff = scale * np.mean(b)
        thresh = np.sqrt(cutoff)
    else:  # Use relative tolerance specified by the user
        cutoff = thresh ** 2
    e = b > cutoff

    return b


###################################################
#
#   Local Function : parse_inputs
#
def parse_inputs(*args):
    # OUTPUTS:
    #   I      Image Data
    #   Method Edge detection method
    #   Thresh Threshold value
    #   Sigma  standard deviation of Gaussian
    #   H      Filter for Zero-crossing detection
    #   kx,ky  From Directionality vector

    # Defaults
    I = args[0]
    Method = 'sobel'
    Direction = 'both'
    Thinning = False

    methods = ['canny', 'canny_old', 'prewitt', 'sobel', 'marr-hildreth', 'log', 'roberts', 'zerocross']
    directions = ['both', 'horizontal', 'vertical']
    options = ['thinning', 'nothinning']

    # Now parse the nargin-1 remaining input arguments

    # First get the strings - we do this because the interpretation of the
    # rest of the arguments will depend on the method.
    nonstr = []  # ordered indices of non-string arguments
    for i in range(1, len(args)):
        if isinstance(args[i], str):
            str_val = args[i].lower()
            j = [idx for idx, val in enumerate(methods) if val == str_val]
            k = [idx for idx, val in enumerate(directions) if val == str_val]
            l = [idx for idx, val in enumerate(options) if val == str_val]
            if j:
                Method = methods[j[0]]
                if Method == 'marr-hildreth':
                    raise ValueError('EDGE(I, ''marr-hildreth'', ...) is not supported in Python')
            elif k:
                Direction = directions[k[0]]
            elif l:
                Thinning = True if options[l[0]] == 'thinning' else False
            else:
                raise ValueError('Invalid input string: ' + args[i])
        else:
            nonstr.append(i)

    # Now get the rest of the arguments
    # Thresh, Sigma, H, kx, ky = parse_non_string_inputs_edge(args, Method, Direction, nonstr)
    # 找不到该函数的源码
    Thresh = np.array([])
    Sigma = 2
    H = np.array([])
    kx = 1
    ky = 1

    return I, Method, Thresh, Sigma, Thinning, H, kx, ky


def smooth_gradient(I, sigma):
    # Determine filter length
    filter_extent = np.ceil(4 * sigma)
    x = np.arange(-filter_extent, filter_extent + 1)

    # Create 1-D Gaussian Kernel
    c = 1 / (np.sqrt(2 * np.pi) * sigma)
    gauss_kernel = c * np.exp(-(x**2) / (2 * sigma**2))

    # Normalize to ensure kernel sums to one
    gauss_kernel /= np.sum(gauss_kernel)

    # Create 1-D Derivative of Gaussian Kernel
    deriv_gauss_kernel = np.gradient(gauss_kernel)

    # Normalize to ensure kernel sums to zero
    deriv_gauss_kernel[deriv_gauss_kernel > 0] /= np.sum(deriv_gauss_kernel[deriv_gauss_kernel > 0])
    deriv_gauss_kernel[deriv_gauss_kernel < 0] /= -np.sum(deriv_gauss_kernel[deriv_gauss_kernel < 0])

    # Compute smoothed numerical gradient of image I along x (horizontal) direction
    GX = convolve(I, gauss_kernel[np.newaxis, :], mode='constant', cval=0.0)
    GX = convolve(GX, deriv_gauss_kernel[:, np.newaxis], mode='constant', cval=0.0)

    # Compute smoothed numerical gradient of image I along y (vertical) direction
    GY = convolve(I, gauss_kernel[:, np.newaxis], mode='constant', cval=0.0)
    GY = convolve(GY, deriv_gauss_kernel[np.newaxis, :], mode='constant', cval=0.0)

    return GX, GY


def select_thresholds(thresh, mag_grad, percent_of_pixels_not_edges, threshold_ratio):
    m, n = mag_grad.shape

    # Select the thresholds
    if thresh is None:
        counts, edges = np.histogram(mag_grad.flatten(), bins=64, range=[0, 1])
        cumulative_counts = np.cumsum(counts)
        high_thresh_index = np.argmax(cumulative_counts > percent_of_pixels_not_edges * m * n)
        high_thresh = edges[high_thresh_index] / 64
        low_thresh = threshold_ratio * high_thresh
    elif len(thresh) == 1:
        high_thresh = thresh[0]
        if high_thresh >= 1:
            raise ValueError('Threshold must be less than 1')
        low_thresh = threshold_ratio * high_thresh
    elif len(thresh) == 2:
        low_thresh, high_thresh = thresh
        if low_thresh >= high_thresh or high_thresh >= 1:
            raise ValueError('Thresholds out of range')
    else:
        raise ValueError('Invalid threshold input')

    return low_thresh, high_thresh


def thinAndThreshold(dx, dy, magGrad, lowThresh, highThresh):
    E = canny_find_local_maxima(3, dx, dy, magGrad)  # 暂时默认direction=3

    if np.any(E):
        # Find strong edges
        strong_edges = (magGrad > highThresh) & E

        # Label connected components in strong edges
        labeled_strong_edges, num_labels = label(strong_edges)

        # Initialize H as a boolean array of the same size as E
        H = np.zeros_like(E, dtype=bool)

        if num_labels > 0:
            # Loop through each labeled region
            for label_idx in range(1, num_labels + 1):
                # Find coordinates of strong edges for the current label
                coords = np.column_stack(np.where(labeled_strong_edges == label_idx))

                # Convert coordinates to row and column indices
                r_strong, c_strong = coords[:, 0], coords[:, 1]

                # Select the region in E corresponding to the current label
                region_mask = labeled_strong_edges == label_idx
                region_E = E & region_mask

                # Use bwselect to connect weak edges in the current region
                region_H = binary_erosion(region_E, structure=np.ones((3, 3)))
                region_H = binary_dilation(region_H, structure=np.ones((3, 3)))

                # Update H with the current region
                H = H | region_H

        else:
            H = np.zeros_like(E, dtype=bool)

    else:
        H = np.zeros_like(E, dtype=bool)

    return H


def canny_find_local_maxima(direction, ix, iy, mag):
    # 函数声明和MATLAB里的有点不一样，可能因为版本不同
    # ix - 水平方向的图像梯度
    # iy - 垂直方向的图像梯度
    # mag - 梯度幅值（矩阵）
    #
    # 梯度方向有4种
    # X代表某个像素，共有8个方向，每个方向相隔45度
    # 我们只用关注其中4个方向，其余4个是对称的
    ix = ix.T
    iy = iy.T
    mag = mag.T
    n, m = mag.shape

    # Find the indices of all points whose gradient (specified by the
    # vector (ix,iy)) is going in the direction we're looking at.
    if direction == 1:
        idx = np.where((iy <= 0) & (ix > -iy) | (iy >= 0) & (ix < -iy))
    elif direction == 2:
        idx = np.where((ix > 0) & (-iy >= ix) | (ix < 0) & (-iy <= ix))
    elif direction == 3:
        idx = np.where((ix <= 0) & (ix > iy) | (ix >= 0) & (ix < iy))
    elif direction == 4:
        idx = np.where((iy < 0) & (ix <= iy) | (iy > 0) & (ix >= iy))

    idx = idx[0] * m + idx[1] + 1
    # Exclude the exterior pixels
    if idx.size != 0:
        v = np.mod(idx, m)
        ext_idx = (v == 1) | (v == 0) | (idx <= m) | (idx > (n - 1) * m)
        idx = np.delete(idx, np.where(ext_idx))

    ixv = ix.flatten()[idx - 1]
    iyv = iy.flatten()[idx - 1]
    gradmag = mag.flatten()[idx - 1]

    # Do the linear interpolations for the interior pixels
    if direction == 1:
        d = np.abs(iyv / ixv)
        gradmag1 = mag.flatten()[idx + m - 1] * (1 - d) + mag.flatten()[idx + m - 2] * d
        gradmag2 = mag.flatten()[idx - m - 1] * (1 - d) + mag.flatten()[idx - m] * d
    elif direction == 2:
        d = np.abs(ixv / iyv)
        gradmag1 = mag.flatten()[idx - 2] * (1 - d) + mag.flatten()[idx + m - 2] * d
        gradmag2 = mag.flatten()[idx] * (1 - d) + mag.flatten()[idx - m] * d
    elif direction == 3:
        d = np.abs(ixv / iyv)
        f_mag = mag.flatten()
        gradmag1 = mag.flatten()[idx - 2] * (1 - d) + mag.flatten()[idx - m - 2] * d
        gradmag2 = mag.flatten()[idx] * (1 - d) + mag.flatten()[idx + m] * d
    elif direction == 4:
        d = np.abs(iyv / ixv)
        gradmag1 = mag.flatten()[idx - m - 1] * (1 - d) + mag.flatten()[idx - m - 2] * d
        gradmag2 = mag.flatten()[idx + m - 1] * (1 - d) + mag.flatten()[idx + m] * d

    idx_local_max = idx[(gradmag >= gradmag1) & (gradmag >= gradmag2)]
    return idx_local_max

