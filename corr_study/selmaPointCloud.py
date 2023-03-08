import open3d as o3d
import seaborn as sns
import numpy as np
from . import voxels
from . import correlation
import sparse
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from tqdm import trange

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

        coordinates = coordinates[:, np.all(coordinates >= 0, axis=0)]
        
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
    
    def compare_using_dbscan(self, target, eps, min_samples):
        
        def compute_oddly_normalized_distance_unilateral(x, y):
            return np.sqrt(np.sum((x - y)**2, axis=1) / np.sqrt(np.sum(x**2, axis=1) * np.sum(y**2)))
        
        def my_dbscan(data, eps, min_samples):
            neighs = [None, ] * data.shape[0]

            def compute_neighbours(data, point_idx, eps):
                if neighs[point_idx] is None:
                    neighs[point_idx] = np.where(compute_oddly_normalized_distance_unilateral(data, data[point_idx,:]) < eps)[0]
                return neighs[point_idx]
            
            def expand_cluster(data, labels, point_idx, cluster_id, eps, min_samples):
                seeds_idxs = compute_neighbours(data, point_idx, eps)
                if len(seeds_idxs) < min_samples:
                    labels[point_idx] = -1
                    return False
                else:
                    labels[seeds_idxs] = cluster_id
                    seeds_idxs = np.delete(seeds_idxs, np.where(seeds_idxs == point_idx))
                    while len(seeds_idxs) != 0:
                        current_idx = seeds_idxs[0]
                        results_idxs = compute_neighbours(data, current_idx, eps)
                        if len(results_idxs) >= min_samples:
                            for index in results_idxs:
                                if labels[index] < 0:
                                    if labels[index] == -2:
                                        seeds_idxs = np.append(seeds_idxs, index)
                                    labels[index] = cluster_id
                        seeds_idxs = np.delete(seeds_idxs, np.where(seeds_idxs == current_idx))
                    return True


            cluster_id = 0
            labels = np.zeros(data.shape[0]) - 2
            for i in trange(data.shape[0]):
                if labels[i] == -2:
                    if expand_cluster(data, labels, i, cluster_id, eps, min_samples):
                        cluster_id += 1
            return labels.astype(int) 

        def dbcompare(labels_first, first_pointcloud, second_pointcloud, eps, min_samples):
            neighs = [None, ] * second_pointcloud.shape[0]

            def compute_neighbours(data, point_idx, eps):
                if neighs[point_idx] is None:
                    neighs[point_idx] = np.where(compute_oddly_normalized_distance_unilateral(data, data[point_idx,:]) < eps)[0]
                return neighs[point_idx]
            
            def expand_cluster(data, labels, point_idx, cluster_id, eps, min_samples):
                seeds_idxs = compute_neighbours(data, point_idx, eps)
                if len(seeds_idxs) < min_samples:
                    labels[point_idx] = -1
                    return False
                else:
                    labels[seeds_idxs] = cluster_id
                    seeds_idxs = np.delete(seeds_idxs, np.where(seeds_idxs == point_idx))
                    while len(seeds_idxs) != 0:
                        current_idx = seeds_idxs[0]
                        results_idxs = compute_neighbours(data, current_idx, eps)
                        if len(results_idxs) >= min_samples:
                            for index in results_idxs:
                                if labels[index] < 0:
                                    if labels[index] == -2:
                                        seeds_idxs = np.append(seeds_idxs, index)
                                    labels[index] = cluster_id
                        seeds_idxs = np.delete(seeds_idxs, np.where(seeds_idxs == current_idx))
                    return True
                
            labels_second = np.zeros(second_pointcloud.shape[0]) - 1
                
            for i in trange(len(np.unique(labels_first))-1):
                _, c = np.unique(labels_first, return_counts=True)
                centroid_first = first_pointcloud[labels_first == i].mean(axis=0)
                closest_point = np.argsort(np.sum((second_pointcloud - centroid_first)**2, axis=1))
                for index_closest in closest_point[:int(c[i + 1]/2)]:
                    if labels_second[index_closest] == -1:
                        expand_cluster(second_pointcloud, labels_second, index_closest, i, eps, min_samples)
            # for i in trange(len(np.unique(labels_first))-1):
            #     _, c = np.unique(labels_first, return_counts=True)
            #     for _ in range(int(c[i + 1]/50)):
            #         ww = np.where(labels_first == i)[0]
            #         if len(ww)!= 0:
            #             index_closest = np.random.choice(ww)
            #             if labels_second[index_closest] == -1:
            #                 expand_cluster(second_pointcloud, labels_second, index_closest, i, eps, min_samples)
            return labels_second
        
        def compute_db_mse(data_first, data_second, labels_first, labels_second):
            mse = 0
            for i in range(len(np.unique(labels_first))-1):
                centroid_first = data_first[labels_first == i].mean(axis=0)
                centroid_second = data_second[labels_second == i].mean(axis=0)
                if not np.isnan(centroid_second).any():
                    mse += np.sum((centroid_first - centroid_second)**2)/np.sum((centroid_first)**2)
            return mse
        
        mses = []
        labels_original = my_dbscan(self.data, eps, min_samples)
        labels_prev = labels_original
        pc_prev = self.data
        for pc_data in target:
            actual_labels = dbcompare(labels_prev, pc_prev, pc_data, eps, min_samples)
            mses.append(compute_db_mse(self.data, pc_data, labels_original, actual_labels))
            labels_prev = actual_labels
            pc_prev = pc_data
        return mses
