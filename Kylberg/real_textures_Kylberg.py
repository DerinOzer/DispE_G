import cv2
import numpy as np

from DispE_G import connected_adjacency, DispE_G

images = ['Kylberg/blanket1_sample_image.png', 'Kylberg/canvas1_sample_image.png',
          'Kylberg/ceiling1_sample_image.png', 'Kylberg/floor1_sample_image.png',
          'Kylberg/floor2_sample_image.png', 'Kylberg/rice1_sample_image.png',
          'Kylberg/rug1_sample_image.png', 'Kylberg/scarf1_sample_image.png',
          'Kylberg/scarf2_sample_image.png', 'Kylberg/screen1_sample_image.png']

# Different values for m and c can be used.
c = 5
m = 2

for im in images:
    image = cv2.imread(im, 0)
    image = image[:256, :256]
    A = connected_adjacency(image, '8')
    X = np.reshape(image, (1, image.shape[0] * image.shape[1]))
    X = X[0]
    dispersion_entropy = DispE_G(A, X, c, m)
    print(dispersion_entropy)