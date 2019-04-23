from PIL import Image
import time
import math

def compress_and_wb(image_path):
    default_size = 96
    image_ex = Image.open(image_path)
    new_image = Image.new("L", (default_size, default_size))
    pixels = new_image.load()
    region = (image_ex.size[0]/default_size, image_ex.size[1]/default_size)

    for w in range(default_size):
        for h in range(default_size):
            current_region = (w*region[0], h*region[1])
            num = 0

            for i in range(math.floor(current_region[0]), math.ceil(current_region[0] + region[0])):
                for j in range(math.floor(current_region[1]), math.ceil(current_region[1] + region[1])):
                    ppx = image_ex.getpixel((i,j))
                    num += wb_converter(rgb=ppx) #from RGB to L

            num = math.ceil(num/( (math.ceil(current_region[0] + region[0]) - math.floor(current_region[0]))*(math.ceil(current_region[1] + region[1]) - math.floor(current_region[1]))))
            pixels[w, h] = num

    new_image.show()
    new_image.save("out_"+image_path)

def wb_converter(rgb):
    assert len(rgb) == 3, "not rgb tuple"
    return rgb[0] *299/1000 + rgb[1] * 587/1000 + rgb[2] * 114/1000
