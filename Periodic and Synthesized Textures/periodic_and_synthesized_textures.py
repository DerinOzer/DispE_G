import cv2
import numpy as np
from DispE_G import connected_adjacency, DispE_G

I1 = cv2.imread('P1.jpg', 0)
I2 = cv2.imread('S1.jpg', 0)
I3 = cv2.imread('P2.jpg', 0)
I4 = cv2.imread('S2.jpg', 0)
I5 = cv2.imread('P3.jpg', 0)
I6 = cv2.imread('S3.jpg', 0)
I7 = cv2.imread('P4.jpg', 0)
I8 = cv2.imread('S4.jpg', 0)
I9 = cv2.imread('P5.jpg', 0)
I10 = cv2.imread('S5.jpg', 0)
I11 = cv2.imread('P6.jpg', 0)
I12 = cv2.imread('S6.jpg', 0)
I = [I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11, I12]
# 6 periodic textures and their synthesized equivalents.

# Different values of m and c can be chosen.
c = 4
m = 3

for image in I:
    image = image[:230, :230]

    A = connected_adjacency(image, '8')
    X = np.reshape(image, (1, image.shape[0] * image.shape[1]))
    X = X[0]

    dispersion_entropy = DispE_G(A, X, c, m)
    print(dispersion_entropy)
