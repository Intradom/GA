from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

pic_path = str(sys.argv[1])
A = misc.imread(pic_path, flatten=True)
plt.imshow(np.abs(A-255), cmap="gray")
print(np.max(-A))
plt.show()