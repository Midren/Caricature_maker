from PIL import Image
from math import floor, ceil


def compress_and_wb(image_path):
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

                    num += wb_converter(rgb=ppx)  # from RGB to L

            num = ceil(num / ((ceil(n_region[0]) - floor(c_region[0])) * (ceil(n_region[1]) - floor(c_region[1]))))
            pixels[w, h] = num

    new_image.save("out_" + image_path)


def wb_converter(rgb):
    if len(rgb) == 3:
        return rgb[0] * 299 / 1000 + rgb[1] * 587 / 1000 + rgb[2] * 114 / 1000


compress_and_wb("hurley-square.jpg")
