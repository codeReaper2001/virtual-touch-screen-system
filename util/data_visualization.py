import matplotlib.pyplot as plt
from typing import List
import numpy as np

def show_dataline_img(dataset: List[List[float]]):
    for dataline in dataset:
        one_record = np.array(dataline)
        one_record = one_record * 1000
        
        x = one_record[::3]
        y = one_record[1::3]
        z = one_record[2::3]

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x, y, z, color="red")
        ax.set_aspect('equal')

        plt.xlabel('x')
        plt.ylabel('y')
    plt.show()