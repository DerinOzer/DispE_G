import cv2
import numpy as np
from DispE_G import connected_adjacency, DispE_G

images = ['Brodatz/D5.png', 'Brodatz/D15.png', 'Brodatz/D30.png', 'Brodatz/D36.png',
          'Brodatz/D45.png', 'Brodatz/D75.png', 'Brodatz/D93.png', 'Brodatz/D95.png',
          'Brodatz/D102.png']

# Different values for m and c can be used.
c = 4
m = 2

for im in images:
    image = cv2.imread(im, 0)
    image = image[:128, :128]
    A = connected_adjacency(image, '8')
    X = np.reshape(image, (1, image.shape[0] * image.shape[1]))
    X = X[0]
    dispersion_entropy = DispE_G(A, X, c, m)
    print(dispersion_entropy)
