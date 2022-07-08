import numpy as np
import matplotlib.pyplot as plt
import sys

if len(sys.argv) != 2:
    print("Usage: python3 plot_distogram.py [npy_file]\n")
    exit()

a = np.load(sys.argv[1])
plt.imshow(a, cmap='hot', interpolation='nearest')
plt.show()
