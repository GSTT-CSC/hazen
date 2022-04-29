import pydicom
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# get the matrix intensity from the pydicom
dataset = pydicom.dcmread('/Users/lce21/Documents/GitHub/hazen/tests/data/resolution/eastkent/256_sag.IMA')
pixels=dataset.pixel_array
#pixels=blur_image(pixels,1)
#print(pixels)
#pitch = dataset.PixelSpacing
#print(pitch)

#Bluring the image allows to test if the MTF changes with different image resolutions
def blur_image(img,blur_value):
    kernel = np.ones((blur_value,blur_value),np.float32)/(blur_value)**2
    img = np.array(img)
    img = cv.filter2D(img,-1,kernel)
    return img

def polynomialfit(data, order):
    '''
    calculate the polynomial fit of an input for a defined degree
    '''
    x, y = range(len(data)), data
    coefficients = np.polyfit(x, y, order)
    return np.polyval(coefficients, x)


# find edge indexes and values
central_col=pixels[120:140,100:120]
plt.imshow(central_col)
plt.show()
columns = central_col.shape[0]
der2_index = []
max_values=[]
max_indexes=[]
for i in range(columns):
    derivative=np.gradient(central_col[i,1:])
    der2 = np.gradient(derivative)
    der2 = np.round(der2)
    max_value = np.amax(der2)
    max_values.append(max_value)
    max_index = np.where(der2 == np.amax(der2))
    max_indexes.append(max_index)

#find angle through sin
y_edge = len(max_indexes)#y value - height of edge
max_indexes = np.asarray(max_indexes)
x_adj =  max_indexes[-1] - max_indexes[0]
hyp = np.sqrt(y_edge**2+x_adj**2)
angle = np.cos(y_edge /hyp)

#project
esp = []
arr = np.array(central_col)
arr=(arr.flatten())
for i in central_col.flatten():
    x_projection = i/np.tan(angle)
    esp.append(x_projection)
out = np.concatenate(esp).ravel().tolist()
esp = sorted(out)
esp = polynomialfit(esp,20)
plt.plot(esp)
plt.show()

#get LSF
lsf = np.gradient(esp[5:-15])
import numpy as np
plt.plot(lsf)
plt.show()
w=np.hanning(len(lsf));
lsf= lsf*w;
plt.plot(lsf)
plt.show()

#get MTF
from numpy.fft import fftfreq
import matplotlib.pyplot as plt
lsf = np.array(lsf)
n=lsf.size
print(n)
mtf = abs(np.fft.fft(lsf))
norm_mtf = mtf / max(mtf)

#freqs= fftfreq(n, 1/n)
#mask = freqs >= 0
#plt.figure
norm_mtf_smooth=polynomialfit(norm_mtf,3)
#freqs= fftfreq(n, 1)
freqs= np.fft.fftfreq(n, 1/2)
mask = freqs >= 0
plt.plot(freqs[mask],norm_mtf[mask])
plt.show()


