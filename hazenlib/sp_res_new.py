def polynomialfit(data, order):
    '''
    calculate the polynomial fit of an input for a defined degree
    '''
    x, y = range(len(data)), data
    coefficients = np.polyfit(x, y, order)
    return np.polyval(coefficients, x)

import numpy as np
import cv2 as cv
def blur_image(img,blur_value):
    kernel = np.ones((blur_value,blur_value),np.float32)/(blur_value)**2
    img = np.array(img)

    img = cv.filter2D(img,-1,kernel)
    return img

import pydicom
dataset = pydicom.dcmread('/Users/lce21/Documents/GitHub/hazen/tests/data/resolution/eastkent/256_sag.IMA')
pixels=dataset.pixel_array
pixels=blur_image(pixels,7)
print(pixels)
pitch = dataset.PixelSpacing
print(pitch)

from matplotlib import pyplot as plt
#plt.imshow(pixels, interpolation='nearest')
#plt.show()

central_row=pixels[120:160,127]
central_smooth = polynomialfit(central_row,10)
x = np.linspace(0, 60, 40)
plt.figure
plt.plot(x,central_smooth)
plt.show()


lsf = np.gradient(central_row)
plt.plot(lsf)
plt.show()

from numpy.fft import fftfreq
import matplotlib.pyplot as plt
lsf = np.array(lsf)
n=lsf.size
mtf = abs(np.fft.fft(lsf))
norm_mtf = mtf / mtf[0]

#freqs= fftfreq(n, 1/n)
#mask = freqs >= 0
#plt.figure
norm_mtf_smooth=polynomialfit(norm_mtf,5)
freqs= fftfreq(n, 1)
mask = freqs >= 0
plt.plot(freqs[mask],norm_mtf[mask])
plt.show()




