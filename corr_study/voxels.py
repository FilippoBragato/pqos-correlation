import numpy as np
from . import selmaPointCloud
from . import correlation

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

    def compute_correlation(self, target):
        return correlation.comupte_correlation(self.data, target.data)
    
    def compute_correlation_inferring_offset(self, target):
        cc = correlation.comupte_cross_correlation(self.data,target.data)
        mm = np.where(cc == np.max(cc))
        shifting_target = np.roll(target.data, mm[0][0]-target.data.shape[0], axis=0)
        shifting_target = np.roll(shifting_target, mm[1][0]-target.data.shape[1], axis=1)
        shifting_target = np.roll(shifting_target, mm[2][0]-target.data.shape[2], axis=2)
        x = _get_common_index(target.data.shape[0], mm[0][0])
        y = _get_common_index(target.data.shape[1], mm[1][0])
        z = _get_common_index(target.data.shape[2], mm[2][0])
        return correlation.comupte_correlation(self.data[:x,:y,:z], shifting_target[:x,:y,:z])
    
def _get_common_index(shape, index):
    if index > shape/2:
        return index
    else: 
        return shape - index