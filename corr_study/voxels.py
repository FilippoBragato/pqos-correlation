import numpy as np
from . import selmaPointCloud

class Voxels:

    def __init__(self, data:np.ndarray, boundaries:np.ndarray, voxel_size:float) -> None:
        self.data = data
        self.boundaries = boundaries
        self.voxel_size = voxel_size

    def to_PointCloud(self):
        points = np.array(np.where(self.data)).T*self.voxel_size + self.boundaries[:,0]
        return selmaPointCloud.SelmaPoinCloud(points)
    
    def visualize(self):
        self.to_PointCloud().visualize()