import itertools
from math import *
import numpy as np
from numpy import std, mean
import scipy.sparse as s


def connected_adjacency(image, connect, patch_size=(1, 1)):
    """
        Creates an adjacency matrix from an image where nodes are considered adjacent
        based on 4-connected or 8-connected pixel neighborhoods.

        :param image: 2 or 3 dim array
        :param connect: string, either '4' or '8'
        :param patch_size: tuple (n,m) used if the image will be decomposed into
                       contiguous, non-overlapping patches of size n x m. The
                       adjacency matrix will be formed from the smaller sized array
                       e.g. original image size = 256 x 256, patch_size=(8, 8),
                       then the image under consideration is of size 32 x 32 and
                       the adjacency matrix will be of size
                       32**2 x 32**2 = 1024 x 1024
        :return: adjacency matrix as a sparse matrix (type=scipy.sparse.csr.csr_matrix)
    """
    r, c = image.shape[:2]
    r = int(r / patch_size[0])
    c = int(c / patch_size[1])

    if connect == '4':
        # constructed from 2 diagonals above the main diagonal
        d1 = np.tile(np.append(np.ones(c - 1), [0]), r)[:-1]
        d2 = np.ones(c * (r - 1))
        upper_diags = s.diags([d1, d2], [1, c])
        return (upper_diags + upper_diags.T).toarray()

    elif connect == '8':
        # constructed from 4 diagonals above the main diagonal
        d1 = np.tile(np.append(np.ones(c - 1), [0]), r)[:-1]
        d2 = np.append([0], d1[:c * (r - 1)])
        d3 = np.ones(c * (r - 1))
        d4 = d2[1:-1]
        upper_diags = s.diags([d1, d2, d3, d4], [1, c - 1, c, c + 1])
        return (upper_diags + upper_diags.T).toarray()


def map_X_to_Z_with_NCDF(X, c):
    """
        Maps vector of signals (X) to the embedding vector (Z).

        :param X: vector of signals of each node
        :param c: number of classes
        :return: embedding vector Z of same size as the vector X
    """
    mu = mean(X)
    sigma = std(X)
    Y = []
    Z = []
    for j in range(0, len(X)):
        x = X[j]
        # The normal cumulative distribution function (NCDF) is used as the mapping function.
        ncdf = 0.5 * (1 + erf((x - mu) / (sqrt(2) * sigma)))
        Y.append(ncdf)
        Z.append(round(c * ncdf + 0.5))
    return Z


def map_X_to_Z_with_brain_structure_ratios(X, c):
    """
           Maps vector of signals (X) to the embedding vector (Z). This function is used
           solely in the case of the medical application proposed in this study.

           :param X: vector of signals of each node/brain structure
           :param c: number of classes
           :return: embedding vector Z of same size as the vector X
    """
    Z = []
    Y = [0, 0, 0, 0, 0, 0, 0, 0]
    # There are 8 brain structures.
    for i in range(0, 8):
        # (i)th brain sructure is symmetrical to the (i+1)th brain structure.
        if (i % 2 == 0):
            t1 = X[i] / (X[i] + X[i + 1])
            t2 = X[i + 1] / (X[i] + X[i + 1])
            Y[i] = t1
            Y[i + 1] = t2
    for j in range(0, len(X)):
        Z.append(round(c * Y[j] + 0.5))
    return Z


def map_X_to_Z_with_edge_attributes(X, W, c):
    """
           Maps vector of signals (X) to the embedding vector (Z) while taking into consideration
           the weighted adjacency matrix.

           :param X: vector of signals of each node/brain structure
           :param W: weighted adjacency matrix
           :param c: number of classes
           :return: embedding vector Z of same size as the vector X
    """
    z = []
    for j in range(0, len(X)):
        neighbors = np.where(W[j] != 0)[0]
        sum = 0
        for n in neighbors:
            sum += W[j][n]
        m = sum/len(neighbors)
        z.append(round(c * X[j] * m + 0.5))
    return z


def determine_patterns(A, Z, m=2):
    """
        Determines patterns using the adjacency matrix and the newly constructed
        embedding vectors (Z).

        :param A: adjacency matrix of the graph
        :param Z: embedding vector
        :param m: embedding dimension
        :return: list of patterns extracted from the graph
    """
    patterns = []
    A = np.triu(A)
    for i in range(0, len(Z)):
        root = [[i]]

        for em_dim in range(2, m + 1):
            root_temp = []
            for r in root:
                neighbors = np.where(A[r[-1]] != 0)[0]
                temp = []
                temp.append(r.copy())
                temp = temp[0]
                # Putting each neighbour once next to the starting node
                for n in neighbors:
                    temp.append(n)
                    root_temp.append(temp.copy())
                    temp.remove(n)
            # 'root' contains indexes but not z values
            root = root_temp

        for x in range(0, len(root)):
            temp = []
            for t in range(0, m):
                temp.append(Z[root[x][t]])
            # Extracting the z values using the indexes in 'root' to construct the patterns.
            patterns.append(temp)
    return patterns


def possible_patterns(c, m=2):
    """
        Generates all possible patterns for the specific combination of 'c' and 'm'.

        :param c: number of classes
        :param m: embedding dimension
        :return: list of all possible patterns for the given values of 'c' and 'm'
    """
    patterns = []
    for x in range(1, c + 1):
        patterns.append(x)
    possible_patterns = list(itertools.product(patterns, repeat=m))
    patterns = []
    for i in range(1, c + 1):
        temp_pat_i = []
        for p in possible_patterns:
            if p[0] == i:
                temp_pat_i.append(np.array(p))
        patterns.append(temp_pat_i)
    # The parameter 'patterns' is manipulated in the form a matrix
    # to increase computational speed in the next step.
    # The (i)th line of the matrix contains the patterns starting with 'i'.
    return patterns


def frequency_of_patterns(patterns, c, m=2):
    """
        Calculates the partial frequency (the number of times each pattern presents itself)
        of each pattern.

        :param patterns: list of determined patterns
        :param c: number of classes
        :param m: embedding dimension
        :return: a list containing partial frequencies
    """
    patterns = np.array(patterns)
    pattern_list = possible_patterns(c, m)
    # The list of frequencies is initialized.
    frequency = [0] * (c ** m)
    for i in range(0, len(patterns)):
        tempP = patterns[i][0]
        temp = pattern_list[tempP - 1]
        # Since the possible patterns are manipulated in the form of a matrix.
        # We propose a 'cut' parameter to cut ahead in the possible pattern list.
        # The parameter 'cut' is an index telling us from which point in the possible
        # pattern list we should be looking.
        cut = (len(temp) / c) * (patterns[i][1] - 1)
        temp = temp[int(cut):]
        for j in range(0, len(pattern_list[0])):
            if (temp[j] == patterns[i]).all():
                frequency[(tempP - 1) * len(pattern_list[0]) + j + int(cut)] += 1
                break
    return frequency


def DispE_G(A, X, c, m=2):
    """
        Calculates the dispersion entropy on irregular graph signals.

        :param A: adjacency matrix
        :param X: vector of signals on nodes
        :param c: number of classes
        :param m: embedding dimension
        :return: dispersion entropy value
    """
    Z = map_X_to_Z_with_NCDF(X,c)
    patterns = determine_patterns(A, Z, m)
    frequencies = frequency_of_patterns(patterns, c, m)
    add = 0
    for i in range(0, len(frequencies)):
        f = frequencies[i]
        sum_of_frequencies = sum(frequencies)
        if (f != 0):
            ln = np.log(f / sum_of_frequencies)
            t = - (f / sum_of_frequencies) * ln
            add += t
    return round(add / np.log(c ** m), 6)

def DispE_G_with_edge_attributes(W, X, c, m=2):
    """
        Calculates the dispersion entropy for weighted graphs.

        :param W: weighted adjacency matrix
        :param X: vector of signals on nodes
        :param c: number of classes
        :param m: embedding dimension
        :return: dispersion entropy value
    """
    Z = map_X_to_Z_with_edge_attributes(X, W, c)
    patterns = determine_patterns(W, Z, m)
    frequencies = frequency_of_patterns(patterns, c, m)
    add = 0
    for i in range(0, len(frequencies)):
        f = frequencies[i]
        sum_of_frequencies = sum(frequencies)
        if (f != 0):
            ln = np.log(f / sum_of_frequencies)
            t = - (f / sum_of_frequencies) * ln
            add += t
    return round(add / np.log(c ** m), 6)


def DispE_G_for_medical_application(A, X, c, m=2):
    """
        Calculates the dispersion entropy on graphs extracted from brain MRIs.

        :param A: adjacency matrix
        :param X: vector of signals on nodes/brain structures
        :param c: number of classes
        :param m: embedding dimension
        :return: dispersion entropy value
    """
    Z = map_X_to_Z_with_brain_structure_ratios(X, c)
    patterns = determine_patterns(A, Z, m)
    frequencies = frequency_of_patterns(patterns, c, m)
    add = 0
    for i in range(0, len(frequencies)):
        f = frequencies[i]
        sum_of_frequencies = sum(frequencies)
        if (f != 0):
            ln = np.log(f / sum_of_frequencies)
            t = - (f / sum_of_frequencies) * ln
            add += t
    return round(add / np.log(c ** m), 6)


def calculate_entropy_on_multiple_images(images, c, m=2):
    """
        Calculates the mean of dispersion entropy on graph of multiple images.

        :param images: set of images for which dispersion entropy on graph will be calculated
        :param c: number of classes
        :param m: embedding dimension
        :return: the mean of dispersion entropy on graph of given images
    """
    sum = 0
    for im in images:
        A = connected_adjacency(im, '8')
        X = np.reshape(im, (1, im.shape[0] * im.shape[1]))
        X = X[0]

        dispersion_entropy = DispE_G(A, X, c, m)
        sum += dispersion_entropy
    return np.round(sum / len(images), 6)





