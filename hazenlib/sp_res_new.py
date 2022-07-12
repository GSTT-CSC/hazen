import pydicom
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# get the matrix intensity from the pydicom
dataset = pydicom.dcmread('/Users/lce21/Documents/GitHub/hazen/tests/data/resolution/eastkent/256_sag.IMA')
pixels=dataset.pixel_array


pitch = dataset.PixelSpacing



#Bluring the image allows to test if the MTF changes with different image resolutions
def blur_image(img,blur_value):
    kernel = np.ones((blur_value,blur_value),np.float32)/(blur_value)**2
    img = np.array(img)
    img = cv.filter2D(img,-1,kernel)
    return img

#pixels=blur_image(pixels,2)
#print(pixels)

def polynomialfit(data, order):
    '''
    calculate the polynomial fit of an input for a defined degree
    '''
    x, y = range(len(data)), data
    coefficients = np.polyfit(x, y, order)
    return np.polyval(coefficients, x)


# find edge indexes and values
central_col=pixels[120:140,100:120] #indexes and values of edge
plt.imshow(central_col)
plt.show()
columns = central_col.shape[0] #get number of columns in the image
der2_index = []
max_values=[]
max_indexes=[]
for i in range(columns): # get indeces of edge, this is the max value because it is the max value in the graph of the second derivtive
    derivative=np.gradient(central_col[i,1:])
    der2 = np.gradient(derivative)
    der2 = np.round(der2)
    max_value = np.amax(der2)
    max_values.append(max_value)
    max_index = np.where(der2 == np.amax(der2))
    max_indexes.append(max_index)

print('max index', max_indexes[0][0])
print('min', max_indexes[-1][0])
angle = 6.3






#find angle through sin, y value - height of edge
y_edge = len(max_indexes) #this is the height of the image, the number of columns and it's the height of the triangle formed by the edge
max_indexes = np.asarray(max_indexes)
x_adj =  max_indexes[-1] - max_indexes[0] #this is the width of the triangle
hyp = np.sqrt(y_edge**2+x_adj**2)

angle = np.arccos(x_adj/hyp)

print('this is angle',angle)

#project
esp = []
arr = np.array(central_col)
arr=(arr.flatten())
for i in central_col.flatten():
    x_projection = i/np.tan(angle)
    esp.append(x_projection)
out = np.concatenate(esp).ravel().tolist()
esp = sorted(out)


import pandas as pd
#project
esp = []
new_val=[]
final_x_cords = []
x_projections = []
values = []
rev_esp = []
tot=[]
esp = np.empty([0,2])
for column in range(central_col.shape[1]):
    x_projections = []
    for index, value in np.ndenumerate(central_col[::-1,column]):
        x_projection = (index[0]+1)*np.tan(0.2) + (column+1)*np.cos(0.2) #https://www.spiedigitallibrary.org/journals/optical-engineering/volume-57/issue-1/014103/Modified-slanted-edge-method-for-camera-modulation-transfer-function-measurement/10.1117/1.OE.57.1.014103.short?SSO=1
        print(x_projection)
        x_projection = np.round(x_projection,1)
        tot1 = np.array([x_projection,value])
        tot.append(tot1)
    #rev_esp = [(x, y) for x,y in zip(x_projection, values)]
    #final_x_cords.extend(rev_esp)

df = pd.DataFrame(tot, columns = ['key', 'values'])
b=(df.groupby('key').mean()).to_numpy()



#df._to_dict() => {2: 250, 3:: }
plt.plot(b)
plt.show()
#

#out = np.concatenate(esp).ravel().tolist()
#esp = sorted(out)
#plt.plot(out)
#plt.show()

esp = polynomialfit(esp,110)
#plt.plot(esp[5:-15])
#plt.plot(esp)
#plt.title("this")
#plt.show()

#get LSF
#lsf = np.gradient(esp[5:-15])
lsf = np.gradient(esp)
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


#norm_mtf_smooth=polynomialfit(norm_mtf,3)
#freqs= np.fft.fftfreq(n, 1/(8*pitch[0]))
profile_length=len(central_col[0])
freqs= fftfreq(n, profile_length/n)
mask = freqs >= 0
plt.plot(freqs[mask],norm_mtf[mask])
plt.show()





