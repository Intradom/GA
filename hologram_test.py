from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import sys

# Only square pictures

pic_name = str(sys.argv[1])
N = int(sys.argv[2]) # Specifies the side length of the square image to resize to

A = misc.imread(pic_name, mode = 'F')
A = misc.imresize(A, (N, N)) # Original Image

# Dirac Delta Function
B = np.zeros((N, N))
B[int(N / 2), int(N / 2)] = 100

A1 = np.fft.fftshift(np.fft.fft2(A))
I1 = np.multiply(abs(A1), abs(A1)) # Change to intensity to display
B1 = np.fft.fftshift(np.fft.fft2(B))
I2 = np.multiply(abs(B1), abs(B1)) # Change to intensity to display

D1 = A1 + B1
I3 = np.multiply(abs(D1), abs(D1))
I4 = I3 - I1 # Hologram

D2 = np.fft.ifft2(np.fft.fftshift(I4))
I5 = np.multiply(abs(D2), abs(D2)) # Reconstructed image, with intensity

plt.figure(1)
plt.subplot(2, 2, 1)
plt.axis('off')
plt.title('Original')
plt.imshow(A, extent=[0, 1, 0, 1], cmap = 'gray')
plt.subplot(2, 2, 2)
plt.axis('off')
plt.title('FFT')
plt.imshow(I4, extent=[0, 1, 0, 1], cmap = 'gray')
plt.subplot(2, 2, 3)
plt.axis('off')
plt.title('Reconstruction')
plt.imshow(I5, extent=[0, 1, 0, 1], cmap = 'gray')
plt.show()
