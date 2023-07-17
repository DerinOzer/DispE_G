import numpy as np
import csv
import pandas as pd

from DispE_G import DispE_G_for_medical_application

data_name = pd.read_csv("subject_names.csv", header=None)
data_name = data_name.iloc[:, 0]
subject_names = list(data_name)

data_node = pd.read_csv("node_attributes_elongation.csv", header=None)
#data_node = pd.read_csv("node_attributes_volume.csv", header=None)
data_edge = pd.read_csv("edge_attributes.csv", header=None)

c = 4
m = 5

for i in range(0, len(data_node)):
    if (i != 12): # This line should be commented while working with volumes as the node attribute function.
        X = []
        A = []
        # We know that there are 8 nodes in the graph
        for j in range(0, 8):
            X.append(data_node.loc[i][j])
        # 8 nodes mean an adjacency matrix of size 8x8 = 64
        for t in range(0, 64):
            A.append(round(data_edge.loc[j - 1][t], 4))
        A = np.resize(A, [8, 8])


        dispersion_entropy = DispE_G_for_medical_application(A, X, c, m)
        print(subject_names[i], dispersion_entropy)

        l = [subject_names[i], dispersion_entropy]
        with open('results.csv', 'a', newline='\n') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(l)