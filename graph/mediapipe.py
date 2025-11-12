import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools

# MediaPipe Pose has 25 joints (using subset from 33 landmarks)
num_node = 25
self_link = [(i, i) for i in range(num_node)]
inward = [
    # Face
    (1, 0), (2, 0),  # eyes to nose
    (3, 1), (4, 2),  # ears to eyes
    # Upper body
    (5, 0), (6, 0),  # shoulders to nose (approximation of upper body center)
    (7, 5), (9, 7),  # left arm: shoulder -> elbow -> wrist
    (8, 6), (10, 8),  # right arm: shoulder -> elbow -> wrist
    (11, 9), (13, 9),  # left wrist to pinky and index
    (12, 10), (14, 10),  # right wrist to pinky and index
    # Torso
    (15, 5), (16, 6),  # hips from shoulders
    (16, 15),  # right hip from left hip
    # Legs
    (17, 15), (19, 17),  # left leg: hip -> knee -> ankle
    (18, 16), (20, 18),  # right leg: hip -> knee -> ankle
    # Feet
    (21, 19), (23, 19),  # left ankle to heel and foot
    (22, 20), (24, 20),  # right ankle to heel and foot
]

outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    # os.environ['DISPLAY'] = 'localhost:10.0'
    A = Graph('spatial').get_adjacency_matrix()
    for i in A:
        plt.imshow(i, cmap='gray')
        plt.show()
    print(A)

