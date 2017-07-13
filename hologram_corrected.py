from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import sys

# Only square pictures
pic_name = str(sys.argv[1])
N = int(sys.argv[2]) # Specifies the side length of the square image to resize to

A = misc.imread(pic_name, flatten=True)
A = misc.imresize(A, (N, N))

A1 = np.fft.fftshift(np.fft.fft2(A))
I1= abs(A1)

D2 = np.fft.ifft2(np.fft.fftshift(A1))

plt.figure(1)
plt.subplot(2, 2, 1)
plt.axis('off')
plt.title('Original')
plt.imshow(A, extent=[0, 1, 0, 1], cmap = 'gray')
plt.subplot(2, 2, 2)
plt.axis('off')
plt.title('FFT')
plt.imshow(I1, extent=[0, 1, 0, 1], cmap = 'gray')
plt.subplot(2, 2, 3)
plt.axis('off')
plt.title('Reconstruction')
plt.imshow(D2.real, extent=[0, 1, 0, 1], cmap = 'gray')
plt.show()