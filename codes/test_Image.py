import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage

import pytest

class Image:
    def __init__(self):
        pass

    def __showVersion__(self):
        print(cv2.__version__)
        print(np.__version__)
        print(pd.__version__)

    def image_read(self, image, color_flag = 0):
        """
        image: path of the image file
        color_flag: 1, 0, -1 (color, grayscale, unchanged)
        default = 0
        """
        if image is None:
            raise('Invalid path')
        if color_flag < -1 and color_flag > 1:
            raise ('Invalid argument for color_path')

        img = cv2.imread(image, color_flag)
        return img

    def image_show(self, image, title='image'):
        """
        image: image array
        title: title of the window
        """
        cv2.imshow(title,image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

class Preprocess:
    def __init__(self):
        pass
    
    def __bbox__(self, image):

        rows = np.any(image, axis = 1)
        cols = np.any(image, axis = 0)

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        return rmin, rmax, cmin, cmax

    def crop_resize(self, image, size = (128, 128), padding = 16):
        height, width = image.shape[0:2]

        xmin, xmax, ymin, ymax = self.__bbox__(image)

        xmin = xmin - 13 if (xmin > 13) else 0
        ymin = ymin - 10 if (ymin > 10) else 0

        xmax = xmax + 13 if (xmax < width - 13) else width
        ymax = ymax + 13 if (ymax < height - 13) else height

        img = image[ymin: ymax, xmin: xmax]

        lx, ly = xmax-xmin, ymax-ymin
        l = max(lx, ly) + padding

        img =  np.pad(img, [((l-ly)//2, ), ((l-lx)//2, )], mode='constant')
        return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    def normalize_image(self, image, threshold = 255):
        """
        image: image array
        threshold: maximum color value
        """
        
        img = image.copy() // threshold
        return img


    def remove_background(self, image, threshold =  235):
        """
        image: image array 
        threshold: values greater than the threshold to be ignored
        removes white background to black
        """
        image[image > threshold] = 0
        return image

    def resize_image(self, image,size = (64,64)):
        """
        image: image array
        size: desired output image size
        """
        return cv2.resize(image, dsize=size,interpolation=cv2.INTER_AREA)
    
    def rotate_image(self, image, angle):
        """
        image: image array
        angle: angle for rotation
        """
        (h, w) = image.shape[:2]
        center = h/2, w/2

        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (h, w))
        return rotated

    def apply_canny(self, image, minVal = 100, maxval = 100):
        """
        image: image array
        minVal:Minimum intensity gradient
        maxVal: maximum intensity gradient

        """
        return cv2.Canny(image, 100, 200)

    def apply_gaussian_blur(self, image,  kernel_size = (3,3)):
        """
        image: image array
        kernel_size: Gaussian Kernel Size. [height width]. 
        height and width should be odd and can have different values. 
        If ksize is set to [0 0], then ksize is computed from sigma values.
        
        """
        return cv2.GaussianBlur(image, kernel_size, cv2.BORDER_DEFAULT)

