from PIL import Image
from math import floor, ceil, sin, pi


class Compressor:
    default_size = (96, 96)
    constant = 2   #different constant k will corespond to interpolation Lanczos_k

    def __init__(self, path):
        self.image = Image.open(path)
        self.rate = self.image.size[0]/Compressor.default_size[0]
        self.Lanczos_constant = round(self.constant*self.rate)

    def lanczos_resampling(self, X_0, Y_0):
        res = 0
        for i in range(X_0 - self.Lanczos_constant + 1, X_0 + self.Lanczos_constant + 1):
            for j in range(Y_0 - self.Lanczos_constant + 1, Y_0 + self.Lanczos_constant + 1):
                try:
                    current_pixel = self.white_n_black(self.image.getpixel((i, j)))
                except:
                    continue
                res += self.__lanczos_kernel__(X_0 - i)*self.__lanczos_kernel__(Y_0 - j)*current_pixel
        return res

    def __lanczos_kernel__(self, x):
        if x == 0:
            return 1
        if -self.Lanczos_constant <= x < self.Lanczos_constant:
            return self.Lanczos_constant*sin(pi*x)*sin(pi*x/self.Lanczos_constant)/((pi*x)**2)
        else:
            return 0

    def lanczos_execution(self):
        new_image = Image.new("L", Compressor.default_size)
        pixels = new_image.load()
        for i in range(new_image.size[0]):
            for j in range(new_image.size[1]):
                pixels[i, j] = round(self.lanczos_resampling(round(self.rate*(i + 0.5)), round(self.rate*(j + 0.5))))
        new_image.save("result.png")


    def whole_image_into_WB(self):
        new = Image.new("L", self.image.size)
        pixels = new.load()
        for i in range(self.image.size[0]):
            for j in range(self.image.size[1]):
                pixels[i, j] = round(self.white_n_black(self.image.getpixel((i,j))))
        new.save("WB_result.jpg")




    def white_n_black(self, rgb):
        return rgb[0] * 299 / 1000 + rgb[1] * 587 / 1000 + rgb[2] * 114 / 1000






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

    new_image.save("old_result.jpg" )



def WB_converter(rgb):
    if len(rgb) == 3:
        return rgb[0] * 299 / 1000 + rgb[1] * 587 / 1000 + rgb[2] * 114 / 1000
