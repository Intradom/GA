"""

Helper script for generating hologram templates, only transforms one image at a time, only tested with .jpg images
To run: python2 <preceding path to file>/hologram_transform.py <Path to desired image>

Transforms images into their holographic form to use for templates in genetic_hologram_generator.py

Only square pictures, that are 1024x1024
Images outputted in holo_templates directory
Output images should be cropped to be 1024x1024 for direct comparison

"""

from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

from numpy.fft import fft2, ifft2, fftshift
from numpy import zeros

N = 1024
  
def lg(x):
  return np.sign(x) * x #np.log(x)

def limg(a):
  return plt.imshow( lg(a), cmap="gray")
  
pic_path = str(sys.argv[1])
A = misc.imread(pic_path, flatten=True)
A = misc.imresize(A, (N, N))

AA = zeros( (2*N,2*N), A.dtype )
AA[:N,:N] = A
AA[N:,:N] = A[::-1,:]
AA[:N,N:] = A[:,::-1]
AA[N:,N:] = A[::-1,::-1]

FA = fft2(AA)
DA = ifft2( FA.real )

fig=plt.figure(3)
ax=fig.add_subplot(1,1,1)
plt.axis('off')
plt.imshow( lg(fftshift(FA.real)), cmap="gray")
#print( lg(fftshift(FA.real)).shape)
base = os.path.basename(pic_path)
save_name = 'holo_templates/' + os.path.splitext(base)[0] + '_holo.jpg'
fig.savefig(save_name, bbox_inches='tight', pad_inches=0, dpi=N)
#plt.show()