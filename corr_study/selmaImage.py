import matplotlib.pyplot as plt
import numpy as np

class SelmaImage:
    def __init__(self, data:np.ndarray, ground_truth:np.ndarray=None, time_step:int=-1) -> None:
        self.data = data
        self.ground_truth = ground_truth
        self.time_step = time_step

    def __str__(self) -> str:
        return str(self.data)
    
    def visualize(self) -> None:
        plt.imshow(self.data[:,:,[2,1,0]])
        plt.show()