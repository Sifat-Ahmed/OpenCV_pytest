from test_Image import Image, Preprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pytest

img = Image()
prep = Preprocess()

im1 = img.image_read('coins.jpg')
im1 = prep.resize_image(im1, size=(256,256))
#im = prep.rotate_image(im, 180)
#im = prep.apply_canny(im)

im = prep.apply_gaussian_blur(im1)

#print(prep.remove_background(im1))


#print(np.any(im, axis = 0))

#plt.imshow(np.hstack((im1, prep.normalize_image(im1))))
#plt.show()


#img.image_show(prep.normalize_image(im1))
fig = plt.figure(figsize=(10,10))

fig.add_subplot(1,2,1)
plt.imshow(im1, cmap = plt.cm.gray)

fig.add_subplot(1,2,2)
plt.imshow(prep.crop_resize(prep.remove_background(im1), size=(512, 512)), cmap=plt.cm.gray)

plt.show()


#print(np.hstack((im1.shape, prep.normalize_image(im1).shape)))