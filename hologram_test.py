from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import sys

# Only square pictures

pic_name = str(sys.argv[1])
N = int(sys.argv[2]) # Specifies the side length of the square image to resize to

A = misc.imread(pic_name, mode = 'F')
A = misc.imresize(A, (N, N))

A1 = np.fft.fftshift(np.fft.fft2(A))
I1= np.multiply(abs(A1), abs(A1))

D2 = np.fft.ifft2(A1)
I5 = np.multiply(abs(D2), abs(D2))

plt.imshow(A, extent=[0, 1, 0, 1], cmap = 'gray')
plt.show()
plt.imshow(I1, extent=[0, 1, 0, 1], cmap = 'gray')
plt.show()
plt.imshow(I5, extent=[0, 1, 0, 1], cmap = 'gray')
plt.show()
