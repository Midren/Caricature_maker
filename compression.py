import argparse
import matplotlib.pyplot as plt
from scipy import linalg
from skimage import img_as_float, io
import numpy as np
from PIL import Image
import os


class Compress:
    def __init__(self, fileName, k):
        self.fileName = fileName
        self.k = k
        if fileName:
            image = io.imread(fileName)  # reads image file into memory
            print
            "Image dimensions {0}".format(image.shape)
            self.image_matrix = img_as_float(
                image)  # converts image to matrix representation

    def start(self):
        if self.fileName:
            if self.isRGB:
                compressed = self.rgb_compression()
            else:
                compressed = self.monochrome_compression()
            #
            file_name, file_extension = os.path.splitext(self.fileName)
            io.imsave("{0}_output{1}".format(file_name, file_extension),
                      compressed)
            io.imshow(compressed)
            io.show()

    def decompose(self, image_matrix):
        # does the singular value decomposition
        # image_matrix should be mxn
        # S is the diagonal/middle matrix that has the singular values
        # U and V are orthogonal matrices
        U, S, V = linalg.svd(image_matrix)
        return U, S, V

    def rgb_compression(self):
        """
        """
        compressed_red = self.monochrome_compression(
            self.image_matrix[:, :, 0], self.k[0])
        compressed_green = self.monochrome_compression(
            self.image_matrix[:, :, 1], self.k[1])
        compressed_blue = self.monochrome_compression(
            self.image_matrix[:, :, 2], self.k[2])
        # populates matrix with zeros
        compressed_image = np.zeros(self.image_matrix.shape,
                                    self.image_matrix.dtype)
        for i in range(self.image_matrix.shape[0]):
            for j in range(self.image_matrix.shape[1]):
                for k in range(self.image_matrix.shape[2]):
                    val = 0
                    if k == 0:
                        val = compressed_red[i][j]
                    elif k == 1:
                        val = compressed_green[i][j]
                    else:
                        val = compressed_blue[i][j]
                    # values must be between -1.0 and 1.0
                    if val < -1.0:
                        val = -1.0
                    elif val > 1.0:
                        val = 1.0
                    compressed_image[i][j][k] = val
        return compressed_image

    def monochrome_compression(self, image_matrix, k):
        U, S, V = self.decompose(image_matrix)
        rank = len(S)
        if rank < k:
            return image_matrix
        # take columns less than k from U
        truncated_u = U[:, :k]
        # take rows less than k from V
        truncated_v = V[:k, :]
        # build the new S matrix with top k diagnal elements
        truncated_s = np.zeros((k, k), image_matrix.dtype)
        for i in range(k):
            truncated_s[i][i] = S[i]
        print
        "truncated_u shape {0}, truncated_s shape {1}, truncated_v shape {2}".format(
            truncated_u.shape, truncated_s.shape, truncated_v.shape)

        # compressed/truncated matrix
        return np.dot(np.dot(truncated_u, truncated_s), truncated_v)

    def isRGB(self):
        return len(self.image_matrix.shape) > 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SVD and Image compression')
    parser.add_argument('-i', dest='fileName', nargs='?', help='image file')
    parser.add_argument('-k', dest='k', nargs='*', default=['5', '5', '5'],
                        help='compression factor k (default 5)')
    args = parser.parse_args()
    fileName = args.fileName
    k = [5, 5, 5]
    c = Compress("is", k)
    c.start()
# how to run code
# python compression.py -i filename.jpeg -k 400 400 400
# python compression.py -i filename.jpeg -k 300 300 300
# python compression.py -i filename.jpeg -k 200 200 200
# python compression.py -i filename.jpeg -k 100 100 100
