import open3d as o3d
import seaborn as sns
import numpy as np
from . import voxels

class SelmaPoinCloud:
    def __init__(self, data:np.ndarray, ground_truth:np.ndarray=None, time_step:int=-1) -> None:
        self.data = data
        self.ground_truth = ground_truth
        self.time_step = time_step

    def __str__(self) -> str:
        return str(self.data)
    
    def visualize(self) -> None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.data)

        if self.ground_truth is not None:
            palette = sns.color_palette("hsv", n_colors=36)
            get_color = lambda tag:palette[tag%36]
            colors = np.array(np.vectorize(get_color)(self.ground_truth)).T

            pcd.colors = o3d.utility.Vector3dVector(colors)

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.get_render_option().background_color = np.asarray([0, 0, 0])
        vis.run()
        vis.destroy_window()

    def voxelize(self, voxel_dimension:float, boundaries=None) -> voxels.Voxels:
        if boundaries is None:

            min_x = np.min(self.data[:, 0])
            min_y = np.min(self.data[:, 1])
            min_z = np.min(self.data[:, 2])
            max_x = np.max(self.data[:, 0])
            max_y = np.max(self.data[:, 1])
            max_z = np.max(self.data[:, 2])

            boundaries = np.array([[min_x, max_x], [min_y, max_y], [min_z, max_z]])

        # Create the matrix for the voxelization
        vox = np.zeros((int(np.ceil((boundaries[0][1]-boundaries[0][0])/voxel_dimension)), 
                        int(np.ceil((boundaries[1][1]-boundaries[1][0])/voxel_dimension)), 
                        int(np.ceil((boundaries[2][1]-boundaries[2][0])/voxel_dimension))), dtype=int)
        
        for point in self.data:
            x = int(np.floor((point[0]-boundaries[0,0])/voxel_dimension))
            y = int(np.floor((point[1]-boundaries[1,0])/voxel_dimension))
            z = int(np.floor((point[2]-boundaries[2,0])/voxel_dimension))
            if x < vox.shape[0] and y < vox.shape[1] and z < vox.shape[2]:
                vox[x, y, z] = 1 #TODO try += 1
            
        return voxels.Voxels(vox, boundaries, voxel_dimension)
    
