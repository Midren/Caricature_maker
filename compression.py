import cv2

from PIL import Image, ImageFilter
from math import floor, ceil, sin, pi, sqrt, atan2, degrees
from numpy import array, full, zeros
import numpy
import matplotlib


class Compressor:
    default_size = (96, 96)
    constant = 2  # different constant k will corespond to interpolation Lanczos_k
    Sobel_dx_matrix = array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Sobel_dy_matrix = Sobel_dx_matrix.transpose()

    def __init__(self, path):
        self.image = Image.open(path)
        self.array_image = numpy.asarray(self.image)

        self.rate = self.image.size[0] / Compressor.default_size[0]
        self.Lanczos_constant = round(self.constant * self.rate)

    def lanczos_resampling(self, X_0, Y_0):
        res = 0
        for i in range(X_0 - self.Lanczos_constant, X_0 + self.Lanczos_constant + 1):
            for j in range(Y_0 - self.Lanczos_constant, Y_0 + self.Lanczos_constant + 1):
                try:
                    current_pixel = self.white_n_black(self.image.getpixel((i, j)))
                except:
                    continue
                res += self.__lanczos_kernel__(i - X_0) * self.__lanczos_kernel__(j - Y_0) * current_pixel
        return res

    def __lanczos_kernel__(self, x):
        if x == 0:
            return 1
        if -self.Lanczos_constant <= x < self.Lanczos_constant:
            return self.Lanczos_constant * sin(pi * x) * sin(pi * x / self.Lanczos_constant) / ((pi * x) ** 2)
        else:
            return 0

    def lanczos_execution(self):
        new_image = Image.new("L", Compressor.default_size)
        pixels = new_image.load()
        for i in range(self.default_size[0]):
            for j in range(self.default_size[1]):
                pixels[i, j] = round(
                    self.lanczos_resampling(round(self.rate * (i + 0.5)), round(self.rate * (j + 0.5))))
        new_image.save("res.png")

    def whole_image_into_WB(self):
        new = Image.new("L", self.image.size)
        pixels = new.load()
        for i in range(self.image.size[0]):
            for j in range(self.image.size[1]):
                pixels[i, j] = round(self.white_n_black(self.image.getpixel((i, j))))
        new.save("WB_result.jpg")

    def white_n_black(self, rgb):
        try:
            return rgb[0] * 299 / 1000 + rgb[1] * 587 / 1000 + rgb[2] * 114 / 1000
        except:
            return rgb

    def __gradient_kernel__(self, x, y, matrix):
        return matrix[x + 1, y + 1]

    def gradient_convolution(self, X_0, Y_0):
        resx = 0
        resy = 0
        for i in range(X_0 - 1, X_0 + 2):
            for j in range(Y_0 - 1, Y_0 + 2):
                try:
                    current_pixel = self.array_image[i, j]
                    current_pixel = sqrt(current_pixel)  # gamma-correction
                except Exception as e:
                    return (0,0)
                resx += self.__gradient_kernel__(X_0 - i, Y_0 - j, Compressor.Sobel_dx_matrix) * current_pixel
                resy += self.__gradient_kernel__(X_0 - i, Y_0 - j, Compressor.Sobel_dy_matrix) * current_pixel

        return (resx, resy)

    def gradient_execution(self):
        dx = full(self.default_size, .0)
        dy = full(self.default_size, .0)
        for i in range(Compressor.default_size[0]):
            for j in range(Compressor.default_size[1]):
                new = self.gradient_convolution(i, j)

                dx[i, j] = new[0]
                dy[i, j] = new[1]

        magnitude = numpy.sqrt(dx ** 2 + dy ** 2)
        angle = (numpy.arctan2(dy, dx)) * 180 / pi
        a = self.matrix2image(magnitude)
        b = self.matrix2image(angle)
        c = self.matrix2image(dx)
        d = self.matrix2image(dy)
        a.save("a.png")
        b.save("b.png")
        c.save("c.png")
        d.save("d.png")
        return (magnitude, angle)

    def matrix2image(self, mat):
        new = Image.new("L", mat.shape)
        pixels = new.load()
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                pixels[j, i] = int(mat[i, j])
        return new

    def HoG__calculation(self, magnitude_matrix, angle_matrix):
        HoG_arrays = zeros((self.default_size[0] // 8, self.default_size[1] // 8, 9), dtype=float)
        for i in range(self.default_size[0] // 8):
            for j in range(self.default_size[1] // 8):
                norm2 = 0
                for c_i in range(8):
                    for c_j in range(8):
                        cind = (8 * i + c_i, 8 * j + c_j)
                        current = angle_matrix[cind[0], cind[1]]

                        weight = (current % 20) / 20
                        HoG_arrays[i, j, (int(current) // 20) % 9] = weight * magnitude_matrix[cind[0], cind[1]]
                        HoG_arrays[i, j, (1 + int(current) // 20) % 9] = (1 - weight) * magnitude_matrix[
                            cind[0], cind[1]]



        orientations = 9
        s_row, s_col = (96, 96)
        c_row, c_col = (8, 8)
        b_row, b_col = (2,2)
        n_cells_row = int(s_row // c_row)  # number of cells along row-axis
        n_cells_col = int(s_col // c_col)
        n_blocks_row = (n_cells_row - b_row) + 1
        n_blocks_col = (n_cells_col - b_col) + 1
        normalized_blocks = numpy.zeros((n_blocks_row, n_blocks_col,
                                         b_row, b_col, orientations))
        eps = 0.0001

        for r in range(n_blocks_row):
            for c in range(n_blocks_col):
                block = HoG_arrays[r:r + b_row, c:c + b_col, :]
                out = block / numpy.sqrt(numpy.sum(block ** 2) + eps ** 2)
                out = numpy.minimum(out, 0.2)
                out = out / numpy.sqrt(numpy.sum(out ** 2) + eps ** 2)
                normalized_blocks[r, c, :] = out
        return HoG_arrays.ravel()



"""

# Python gradient calculation

# Read image
im = numpy.asarray(Image.open("img/aaa.png"))
im = numpy.float32(im) / 255.0

# Calculate gradient

from skimage import filters
gx = filters.sobel_v(im)
gy = filters.sobel_h(im)
dx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=1)
dy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=1)

mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
mag1, angle1 = cv2.cartToPolar(dx, dy, angleInDegrees=True)




c1 = Compressor("img/aaa.png")
a = c1.gradient_execution()
fffff = (a[0] - mag*numpy.max(a[0])/numpy.max(mag))
qqqqq = (a[1] - angle*numpy.max(a[1])/numpy.max(angle))



h = c1.HoG__calculation(mag, angle)
from skimage.feature import hog
from skimage import draw

im = Image.open("img/aaa.png")
ima = numpy.asarray(im)
q1 = hog(ima, block_norm='L2', visualize=True)

s_row, s_col = (96, 96)
c_row, c_col = (8, 8)
b_row, b_col = (2,2)
orientations = 9

n_cells_row = int(s_row // c_row)  # number of cells along row-axis
n_cells_col = int(s_col // c_col)

radius = min(c_row, c_col) // 2 - 1
orientations_arr  = numpy.arange(orientations)
# set dr_arr, dc_arr to correspond to midpoints of orientation bins
orientation_bin_midpoints = (
        numpy.pi * (orientations_arr + .5) / orientations)
dr_arr = radius * numpy.sin(orientation_bin_midpoints)
dc_arr = radius * numpy.cos(orientation_bin_midpoints)
hog_image = numpy.zeros((s_row, s_col), dtype=float)
for r in range(n_cells_row):
    for c in range(n_cells_col):
        for o, dr, dc in zip(orientations_arr, dr_arr, dc_arr):
            centre = tuple([r * c_row + c_row // 2,
                            c * c_col + c_col // 2])
            rr, cc = draw.line(int(centre[0] - dc),
                               int(centre[1] + dr),
                               int(centre[0] + dc),
                               int(centre[1] - dr))
            hog_image[rr, cc] += h[r, c, o]

from matplotlib import pyplot
w = h.ravel()

pyplot.imsave("bil.png", hog_image, cmap='gist_gray')
pyplot.imsave("blaaa.png", q1[1], cmap='gist_gray')

s = 0
for i in range(1296):
    s += q1[0][i] - w[i]
    print(q1[0][i], w[i])
print(s)
"""