from PIL import Image
from math import floor, ceil, sin, pi, sqrt, atan2, degrees
from numpy import array, full, zeros
import numpy
import matplotlib.pyplot as plt
import random

class Compressor:
    default_size = (96, 96)
    constant = 2  # different constant k will corespond to interpolation Lanczos_k
    x = [1, 0, 1]
    Sobel_dx_matrix = array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Sobel_dy_matrix = Sobel_dx_matrix.transpose()

    def __init__(self, path):
        self.image = Image.open(path)
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
        for i in range(new_image.size[0]):
            for j in range(new_image.size[1]):
                pixels[i, j] = round(
                    self.lanczos_resampling(round(self.rate * (i + 0.5)), round(self.rate * (j + 0.5))))
        new_image.save("res.png")
        return new_image

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
                    current_pixel = self.image.getpixel((i, j))
                except Exception as e:
                    continue
                resx += self.__gradient_kernel__(X_0 - i, Y_0 - j, Compressor.Sobel_dx_matrix) * current_pixel
                resy += self.__gradient_kernel__(X_0 - i, Y_0 - j, Compressor.Sobel_dy_matrix) * current_pixel
        magnitude = sqrt(resx ** 2 + resy ** 2)
        angle = abs(degrees(atan2(resx, resy)))
        return (magnitude, angle)

    def gradient_execution(self):
        magnitude = full(self.default_size, .0)
        angle = full(self.default_size, .0)
        for i in range(Compressor.default_size[0]):
            for j in range(Compressor.default_size[1]):
                new = self.gradient_convolution(i, j)
                print(new[0])
                magnitude[i, j] = new[0]
                angle[i, j] = new[1]
        return (magnitude, angle)

    def matrix2image(self, mat):
        new = Image.new("L", self.default_size)
        pixels = new.load()
        for i in range(self.default_size[0]):
            for j in range(self.default_size[1]):
                pixels[i,j] = int(mat[i,j])

        new.save("gradient.png")
    def HoG__calculation(self, magnitude_matrix, angle_matrix):
        HoG_arrays = zeros((self.default_size[0]//8, self.default_size[1]//8, 9), dtype=float)
        for i in range(self.default_size[0] // 8):
            for j in range(self.default_size[1] // 8):
                norm2 = 0
                for c_i in range(8):
                    for c_j in range(8):
                        cind = (8 * i + c_i, 8 * j + c_j)
                        current = angle_matrix[cind[0], cind[1]]

                        weight = (current % 20) / 20
                        print(weight)
                        HoG_arrays[i, j, (int(current) // 20)%9] = weight * magnitude_matrix[cind[0], cind[1]]
                        HoG_arrays[i, j, (1 + int(current - .01) // 20)%9] = (1 - weight) * magnitude_matrix[cind[0], cind[1]]

                for k in range(9):
                    norm2 += HoG_arrays[i, j, k] ** 2

                norm2 = sqrt(norm2)
                for k in range(9):
                    HoG_arrays[i, j, k] /= norm2
        return HoG_arrays.ravel()


"""
        csx = 8
        csy = 8

        max_angle = numpy.pi
        n_cells_y, n_cells_x, nbins = HoG_arrays.shape
        sx, sy = n_cells_x * csx, n_cells_y * csy
        plt.style.use('dark_background')
        plt.close()
        plt.figure()  # figsize=(sx/2, sy/2))#, dpi=1)
        plt.xlim(0, sx)
        plt.ylim(sy, 0)

        center = csx // 2, csy // 2
        b_step = max_angle / nbins


        for i in range(Compressor.default_size[0]//8):
            for j in range(Compressor.default_size[1]//8):
                for k in range(nbins):
                    if HoG_arrays[i, j, k] != 0:
                        length = 1 * HoG_arrays[i, j, k]
                        plt.arrow((center[0] + j * csx) - numpy.cos(b_step * k) * (center[0] - 1),
                                  (center[1] + i * csy) + numpy.sin(b_step * k) * (center[1] - 1),
                                  2 * numpy.cos(b_step * k) * (center[0] - 1), - 2 * numpy.sin(b_step * k) * (center[1] - 1),
                                  width=length, color=  str(1 - length),  # 'black',
                                  head_width=2.2*length, head_length= 2.2*length,
                                  length_includes_head=True)

        plt.show()

        r = HoG_arrays.ravel()
        print(r.max())
        return HoG_arrays.ravel()

    def image_gen(self, size):
        new = Image.new("L", size)
        pixels = new.load()
        for i in range(size[0]):
            for j in range(size[1]):
                pixels[i, j] = random.randint(0, 255)
        new.save("im.png")

c1 = Compressor("im.png")
c1.image_gen((96, 96))
c = Compressor("im.png")
a = c.gradient_execution()
ho = c.HoG__calculation(a[0], a[1])
print(ho)
from skimage.feature import hog
pic = Image.open("im.png")
pix = numpy.array(pic.getdata()).reshape(96, 96)
c.whole_image_into_WB()
res = hog(pix)
print(len(ho), len(res))
print(res)



pic = Image.open("res.png")
pix = numpy.array(pic.getdata()).reshape(96, 96)


c = Compressor("res.png")
a = c.gradient_execution()
c.matrix2image(a[0])
ho = c.HoG__calculation(a[0], a[1])
from skimage.feature import hog

print(ho)
c.whole_image_into_WB()
res = hog(pix, visualize=True)
print(len(ho), len(res[0]))
print(res[0])
plt.imshow(res[1], 'gray')
"""

def OLD_compress_as_mean(image_path):
    default_size = 96
    image_ex = Image.open(image_path)
    new_image = Image.new("L", (default_size, default_size))
    pixels = new_image.load()
    region = (image_ex.size[0] / default_size, image_ex.size[1] / default_size)

    for w in range(default_size):
        for h in range(default_size):

            c_region = (w * region[0], h * region[1])
            n_region = ((w + 1) * region[0], (h + 1) * region[1])
            num = 0

            for i in range(floor(c_region[0]), ceil(n_region[0])):
                for j in range(floor(c_region[1]), ceil(n_region[1])):

                    try:
                        ppx = image_ex.getpixel((i, j))
                    except IndexError:
                        continue

                    num += WB_converter(rgb=ppx)  # from RGB to L

            num = ceil(num / ((ceil(n_region[0]) - floor(c_region[0])) * (ceil(n_region[1]) - floor(c_region[1]))))
            pixels[w, h] = num
    new_image.save("old_result.jpg")


def WB_converter(rgb):
    if len(rgb) == 3:
        return rgb[0] * 299 / 1000 + rgb[1] * 587 / 1000 + rgb[2] * 114 / 1000

# Compressor("img/blet.png").lanczos_execution()
