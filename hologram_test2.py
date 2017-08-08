from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import sys

from numpy.fft import fft2, ifft2, fftshift
from numpy import zeros

# Only square pictures
if 0:
  pic_name = str(sys.argv[1])
  N = int(sys.argv[2]) # Specifies the side length of the square image to resize to

N = 256

if 1:
  pic_name = "circle.jpg"
  A = misc.imread(pic_name, flatten=True)
  A = misc.imresize(A, (N, N))

if 0:
  A = zeros((N,N),float)
  A[(N>>1)-4:(N>>1)+4,(N>>1)-4:(N>>1)+4] = 1

AA = zeros( (2*N,2*N), A.dtype )
AA[:N,:N] = A
AA[N:,:N] = A[::-1,:]
AA[:N,N:] = A[:,::-1]
AA[N:,N:] = A[::-1,::-1]

FA = fft2(AA)
DA = ifft2( FA.real )

def lg(x):
  return np.sign(x) * x #np.log(x)

def limg(a):
  return plt.imshow( lg(a), interpolation="nearest", cmap="gray")

plt.figure(2)
plt.subplot(121); plt.imshow(A)
plt.subplot(122); plt.imshow(DA[:N,:N].real)

plt.figure(1)
plt.subplot(121); limg(fftshift(FA.real))
plt.subplot(122); limg(fftshift(FA.imag))

plt.figure(3) ; limg(fftshift(FA.real))
print(FA.real)
plt.savefig("test.jpg", dpi = 1000)
plt.show()

"""
raise "enough"


A1 = np.fft.fftshift(np.fft.fft2(A))
I1 = abs(A1)**2
D2 = np.fft.ifft2(np.fft.fftshift(I1))

plt.figure(1); clf()
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
plt.imshow(D2.real-A, extent=[0, 1, 0, 1], cmap = 'gray')
colorbar()
plt.show()

l = arange(-64,64)
x,y=meshgrid(l,l)
r = asfarray(x*x+y*y)
f = 1-r/(128.0+x*x*4+y*y)
D3 = np.fft.ifft2(np.fft.fftshift(f*A1))
figure(2); clf(); imshow(D2.real, extent=[0, 1, 0, 1], cmap = 'gray', interpolation='nearest'); colorbar()
figure(3); clf(); imshow(D3.real-A,interpolation='nearest'); colorbar()
"""
