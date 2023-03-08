import open3d as o3d
import seaborn as sns
import numpy as np
from . import voxels
from . import correlation
import sparse
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error

class SelmaPointCloud:
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
            get_color = lambda tag:palette[tag%36] if tag != -1 else (1.0,1.0,1.0)
            colors = np.array(np.vectorize(get_color)(self.ground_truth)).T

            pcd.colors = o3d.utility.Vector3dVector(colors)

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.get_render_option().background_color = np.asarray([0, 0, 0])
        vis.run()
        vis.destroy_window()

    def voxelize(self, voxel_dimension:float, boundaries=None, cumulative=False) -> voxels.Voxels:
        if boundaries is None:

            min_x = np.min(self.data[:, 0])
            min_y = np.min(self.data[:, 1])
            min_z = np.min(self.data[:, 2])
            max_x = np.max(self.data[:, 0])
            max_y = np.max(self.data[:, 1])
            max_z = np.max(self.data[:, 2])

            boundaries = np.array([[min_x, max_x], [min_y, max_y], [min_z, max_z]])

        x = np.floor((self.data[:, 0] - boundaries[0,0]) / voxel_dimension).astype(int)
        y = np.floor((self.data[:, 1] - boundaries[1,0]) / voxel_dimension).astype(int)
        z = np.floor((self.data[:, 2] - boundaries[2,0]) / voxel_dimension).astype(int)


        coordinates = np.unique(np.array([x,y,z]), axis=1)
        shape = (int(np.ceil((boundaries[0][1]-boundaries[0][0])/voxel_dimension)), 
                        int(np.ceil((boundaries[1][1]-boundaries[1][0])/voxel_dimension)), 
                        int(np.ceil((boundaries[2][1]-boundaries[2][0])/voxel_dimension)))

        # print(np.where(coordinates < 0)[1])
        coordinates = coordinates[:, np.all(coordinates >= 0, axis=0)]
        # print(coordinates)
        a = np.where(coordinates >= shape[0])
        b = np.where(coordinates >= shape[1])
        c = np.where(coordinates >= shape[2])

        a = a[1][a[0] == 0]
        b = b[1][b[0] == 1]
        c = c[1][c[0] == 2]

        to_delete = np.concatenate((a,b,c))
        to_delete = np.unique(to_delete)
        
        coordinates = np.delete(coordinates, to_delete, axis=1)

        vox = sparse.COO(coordinates, 1, shape=shape)

        # Create the matrix for the voxelization
        # vox = np.zeros((int(np.ceil((boundaries[0][1]-boundaries[0][0])/voxel_dimension)), 
        #                 int(np.ceil((boundaries[1][1]-boundaries[1][0])/voxel_dimension)), 
        #                 int(np.ceil((boundaries[2][1]-boundaries[2][0])/voxel_dimension))), dtype=np.float64)
        
        # for point in self.data:
        #     x = int(np.floor((point[0]-boundaries[0,0])/voxel_dimension))
        #     y = int(np.floor((point[1]-boundaries[1,0])/voxel_dimension))
        #     z = int(np.floor((point[2]-boundaries[2,0])/voxel_dimension))
        #     if x < vox.shape[0] and y < vox.shape[1] and z < vox.shape[2]:
        #         if cumulative:
        #             vox[x, y, z] += 1
        #         else: 
        #             vox[x, y, z] = 1
            
        return voxels.Voxels(vox, boundaries, voxel_dimension)
    
    def compute_center_of_mass(self, weighted=False):
        weights = None
        if weighted:
            weights = np.sqrt(self.data[:,0]**2 + self.data[:,1]**2 + self.data[:,2]**2)
        return np.average(self.data, weights=weights, axis=0)
    
    def compare_using_voxels(self, target, voxel_size, weighted=False):
        sample_a = self.data - self.compute_center_of_mass(weighted=weighted)
        sample_b = target.data - target.compute_center_of_mass(weighted=weighted)
        min_x = min(np.min(sample_a[:, 0]),np.min(sample_b[:, 0]))
        min_y = min(np.min(sample_a[:, 1]),np.min(sample_b[:, 1]))
        min_z = min(np.min(sample_a[:, 2]),np.min(sample_b[:, 2]))
        max_x = max(np.max(sample_a[:, 0]),np.max(sample_b[:, 0]))
        max_y = max(np.max(sample_a[:, 1]),np.max(sample_b[:, 1]))
        max_z = max(np.max(sample_a[:, 2]),np.max(sample_b[:, 2]))
        boundaries = np.array([[min_x, max_x], [min_y, max_y], [min_z, max_z]])
        vox_b = SelmaPointCloud(sample_b).voxelize(voxel_size, boundaries=boundaries)
        vox_a = SelmaPointCloud(sample_a).voxelize(voxel_size, boundaries=boundaries)
        return vox_a.compute_correlation(vox_b)

    def compare_using_clusters(self, target, number_of_clusters, weighted=False):
        sample_a = self.data - self.compute_center_of_mass(weighted=weighted)
        sample_b = target.data - target.compute_center_of_mass(weighted=weighted)
        kmeans_a = KMeans(n_clusters=number_of_clusters, n_init='auto')
        kmeans_a.fit(sample_a)
        kmeans_b = KMeans(n_clusters=number_of_clusters, init=kmeans_a.cluster_centers_, n_init=1)
        kmeans_b.fit(sample_b)
        return mean_squared_error(kmeans_a.cluster_centers_, kmeans_b.cluster_centers_)
