import numpy as np
import random
import cv2
from DispE_G import connected_adjacency, DispE_G, calculate_entropy_on_multiple_images


def add_sp_noise(image, prob):
    """
        Adds Salt&Pepper noise to an image using the given probability

        :param prob: probability with which the pixels are selected to increase the noise
        :return: image with Salt&Pepper noise
    """
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def add_gaussian_noise(mean, var, image):
    """
        Adds Gaussian noise to an image using the given mean and variation values.

        :param mean: mean value
        :param var: variance value
        :return: image with Gaussian noise
    """
    normalized_image = cv2.normalize(image, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, (normalized_image.shape[0], normalized_image.shape[1]))
    noisy_image = normalized_image + gaussian
    # noisy_image = noisy_image.astype(np.uint8)
    return noisy_image


lena = cv2.imread('Noise/lena.png', 0)
lena = cv2.resize(lena, (256, 256))
cv2.imshow('lena', lena)
cv2.waitKey()

# Different m and c values can be chosen.
c = 3
m = 2

A = connected_adjacency(lena, '8')
X = np.reshape(lena, (1, lena.shape[0] * lena.shape[1]))
X = X[0]
entropy = DispE_G(A, X, c, m)
print('Dispersion entropy of the original image:', entropy)

list_of_p = [0.01, 0.05, 0.09]
mean = []

for p in list_of_p:
    temp_images = []
    # 10 realizations
    for i in range(0, 10):
        lena_gauss = add_gaussian_noise(p, p, lena)
        temp_images.append(lena_gauss)
    ent = calculate_entropy_on_multiple_images(temp_images, c, m)
    mean.append(ent)

print('Image with Gaussian noise')
print('for mu = sigma = 0.01,', mean[0])
print('for mu = sigma = 0.05,', mean[1])
print('for mu = sigma = 0.09,', mean[2])

for p in list_of_p:
    temp_images = []
    # 10 realizations
    for i in range(0, 10):
        lena_sp = add_sp_noise(lena, p)
        temp_images.append(lena_sp)
    ent = calculate_entropy_on_multiple_images(temp_images, c, m)
    mean.append(ent)

print('Image with Salt&Pepper noise')
print('for p = 0.01,', mean[3])
print('for p = 0.05,', mean[4])
print('for p = 0.09,', mean[5])
