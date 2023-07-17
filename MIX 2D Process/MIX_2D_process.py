import numpy as np
import math

from DispE_G import calculate_entropy_on_multiple_images


def generate_image_with_MIX_process(rows, colums, p):
    """
        Generates an image using the MIX 2D process.

        :param rows: number of rows in the image (height of the image)
        :param colums: number of columns in the image (width of the image)
        :param p: probability that defines the irregularity of the image
        :return: an image generated using the MIX 2D process
    """
    image = np.zeros((rows, colums))
    sqrt3 = math.sqrt(3)
    for i in range(0, rows):
        for j in range(0, colums):
            random = np.random.uniform()
            if (random < p):
                image[i][j] = (np.random.uniform() - 0.5) * 2 * sqrt3
            else:
                image[i][j] = math.sin(2 * math.pi * i / 12) + math.sin(2 * math.pi * j / 12)
    return image


# Different values of m and c can be chosen.
c = 2
m = 2
# Generating images with varying sizes from 20x20 to 100x100.
for s in range (2,11):
    s = s*10
    average = []
    # Generating images with different probability values from 0.01 to 0.09.
    for i in range (1,10):
        i = i*0.1
        images_temps = []
        # 20 realizations
        for j in range (1,20):
            images_temps.append(generate_image_with_MIX_process(s, s, i))
        ent = calculate_entropy_on_multiple_images(images_temps,c,m)
        average.append(ent)
    # At the end, we obtain a list of entropy for each image size. The entropies are in the order of increasing probability.
    # The first entropy of the list is the one obtained with p = 0.01 and the last one with p = 0.09.
    print('Size =', s, ' ', average)