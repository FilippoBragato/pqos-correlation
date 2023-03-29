import open3d as o3d
import seaborn as sns
import numpy as np
from . import voxels
from . import correlation
import sparse
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, confusion_matrix
from tqdm import trange
import copy
from scipy.ndimage import label, generate_binary_structure
from scipy.spatial.distance import cdist


NOTHING = -1
CENTER_OF_MASS = 0
ICP_REGISTRATION = 1

NO = 0
PRE_REGISTRATION = 1
POST_REGISTRATION = 2

class SelmaPointCloud:
    def __init__(self, data:np.ndarray, ground_truth:np.ndarray=None, time_step:int=-1) -> None:
        self.data = data
        self.ground_truth = ground_truth
        self.time_step = time_step

    def __str__(self) -> str:
        return str(self.data)

    
    def visualize(self, inferred=False) -> None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.data)

        if not inferred and self.ground_truth is not None:
            palette = sns.color_palette("hsv", n_colors=36)
            get_color = lambda tag:palette[tag%36] if tag != 0 else (1.0,1.0,1.0)
            colors = np.array(np.vectorize(get_color)(self.ground_truth[:,1])).T
            pcd.colors = o3d.utility.Vector3dVector(colors)

        if inferred and hasattr(self, 'isMobile'):
            palette = sns.color_palette("hsv", n_colors=150)
            get_color = lambda tag:palette[tag%150] if tag != 0 else (1.0,1.0,1.0)
            colors = np.array(np.vectorize(get_color)(self.isMobile)).T
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

    def icp_register(self, a:np.ndarray, b:np.ndarray, ignore_center=True, init=None, visualize=False):
        def draw_registration_result(source, target, transformation):
            source_temp = copy.deepcopy(source)
            target_temp = copy.deepcopy(target)
            source_temp.paint_uniform_color([1, 0, 0])
            target_temp.paint_uniform_color([0, 0, 1])
            source_temp.transform(transformation)
            o3d.visualization.draw_geometries([source_temp, target_temp])

        if ignore_center:
            data_a = a[np.sum(a**2, axis=1) > 40]
            data_b = b[np.sum(b**2, axis=1) > 40]
        else:
            data_a = a
            data_b = b
        
        pc_a = o3d.geometry.PointCloud()
        pc_a.points = o3d.utility.Vector3dVector(data_a.data)
        pc_b = o3d.geometry.PointCloud()
        pc_b.points = o3d.utility.Vector3dVector(data_b.data)
        if init is None:
            results = o3d.pipelines.registration.registration_icp(pc_a, pc_b, 1000, 
                                                                  criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=3000))
        else:
            results = o3d.pipelines.registration.registration_icp(pc_a, pc_b, 1000, init=init,
                                                                  criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=3000))
        if visualize:
            draw_registration_result(pc_a, pc_b, results.transformation)
        return results.transformation

    def compare_using_voxels(self, target, voxel_size, weighted=False, mode=CENTER_OF_MASS, crop_street=False, init_transform=None):
        sample_a = self.data
        sample_b = target.data
        if crop_street:
            sample_a = sample_a[sample_a[:,2]>-0.9]
            sample_b = sample_b[sample_b[:,2]>-0.9]
        if mode == CENTER_OF_MASS:
            sample_a = sample_a - self.compute_center_of_mass(weighted=weighted) # TODO remove the street also in this case
            sample_b = sample_b - target.compute_center_of_mass(weighted=weighted)
        elif mode == ICP_REGISTRATION:
            transformation = self.icp_register(sample_a, sample_b, init=init_transform)
            pc_a = o3d.geometry.PointCloud()
            pc_a.points = o3d.utility.Vector3dVector(sample_a)
            pc_a.transform(transformation)
            sample_a = np.asarray(pc_a.points)

        min_x = min(np.min(sample_a[:, 0]),np.min(sample_b[:, 0]))
        min_y = min(np.min(sample_a[:, 1]),np.min(sample_b[:, 1]))
        min_z = min(np.min(sample_a[:, 2]),np.min(sample_b[:, 2]))
        max_x = max(np.max(sample_a[:, 0]),np.max(sample_b[:, 0]))
        max_y = max(np.max(sample_a[:, 1]),np.max(sample_b[:, 1]))
        max_z = max(np.max(sample_a[:, 2]),np.max(sample_b[:, 2]))
        boundaries = np.array([[min_x, max_x], [min_y, max_y], [min_z, max_z]])
        vox_b = SelmaPointCloud(sample_b).voxelize(voxel_size, boundaries=boundaries)
        vox_a = SelmaPointCloud(sample_a).voxelize(voxel_size, boundaries=boundaries)
        if mode == ICP_REGISTRATION:
            return vox_a.compute_correlation(vox_b), transformation
        return vox_a.compute_correlation(vox_b)
    
    def intersection_using_voxels(self, target, voxel_size, crop_street=False):

        sample_a = self.data
        sample_b = target.data
        
        if crop_street:
            sample_a = sample_a[sample_a[:,2]>-0.9]
            sample_b = sample_b[sample_b[:,2]>-0.9]

        min_x = min(np.min(sample_a[:, 0]),np.min(sample_b[:, 0]))
        min_y = min(np.min(sample_a[:, 1]),np.min(sample_b[:, 1]))
        min_z = min(np.min(sample_a[:, 2]),np.min(sample_b[:, 2]))
        max_x = max(np.max(sample_a[:, 0]),np.max(sample_b[:, 0]))
        max_y = max(np.max(sample_a[:, 1]),np.max(sample_b[:, 1]))
        max_z = max(np.max(sample_a[:, 2]),np.max(sample_b[:, 2]))

        boundaries = np.array([[min_x, max_x], [min_y, max_y], [min_z, max_z]])

        vox_b = SelmaPointCloud(sample_b).voxelize(voxel_size, boundaries=boundaries)
        vox_a = SelmaPointCloud(sample_a).voxelize(voxel_size, boundaries=boundaries)

        return vox_a.compute_intersection_size(vox_b)

    def compare_using_clusters(self, target, number_of_clusters, weighted=False, mode=CENTER_OF_MASS, crop_street=False, init_transform=None, visualize=False, return_mse=False):
        sample_a = self.data
        sample_b = target.data
        if crop_street:
            sample_a = sample_a[sample_a[:,2]>-0.9]
            sample_b = sample_b[sample_b[:,2]>-0.9]
        if mode == CENTER_OF_MASS:
            sample_a = sample_a- self.compute_center_of_mass(weighted=weighted)
            sample_b = sample_b - target.compute_center_of_mass(weighted=weighted)
        elif mode == ICP_REGISTRATION:
            transformation = self.icp_register(sample_a, sample_b, init=init_transform, visualize=visualize)
            pc_a = o3d.geometry.PointCloud()
            pc_a.points = o3d.utility.Vector3dVector(sample_a)
            pc_a.transform(transformation)
            sample_a = np.asarray(pc_a.points)
        kmeans_a = KMeans(n_clusters=number_of_clusters, n_init='auto')
        kmeans_a.fit(sample_a)
        kmeans_b = KMeans(n_clusters=number_of_clusters, init=kmeans_a.cluster_centers_, n_init=1)
        kmeans_b.fit(sample_b)
        if return_mse:
            if mode == ICP_REGISTRATION:
                return mean_squared_error(kmeans_a.cluster_centers_, kmeans_b.cluster_centers_), transformation
            else:
                return mean_squared_error(kmeans_a.cluster_centers_, kmeans_b.cluster_centers_)

        if mode == ICP_REGISTRATION:
            return correlation.comupte_correlation(kmeans_a.cluster_centers_, kmeans_b.cluster_centers_), transformation
        return correlation.comupte_correlation(kmeans_a.cluster_centers_, kmeans_b.cluster_centers_)
    
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

    def classify_mobile(self, classifier, threshold=.5):
        points = self.data[self.data[:,2]>-0.9]
        img = np.zeros((1,1024,1024,2))
        img[0,:,:,1] = np.zeros((1024,1024)) - 0.9
        for point in points:
            x_img = int(point[0]/0.16 + 512)
            y_img = int(point[1]/0.16 + 512)
            if x_img>=0 and x_img<1024 and y_img>=0 and y_img<1024:
                img[0, x_img, y_img, 0] += 1
                img[0, x_img, y_img, 1] += point[2]
        mask = img[0,:,:,0] > 0
        img[0,mask,1] = img[0,mask,1] / img[0,mask,0]
        img[0,:,:,1] = (img[0,:,:,1] + 1)/(10 + 1) 
        mask = img[0,:,:,0] < 10
        img[0,mask, 0] /= 10
        img[0,~mask, 0] = 1
        pred = classifier.predict(img, verbose=0) > threshold
        pred = pred[0,:,:,0]
        structure = generate_binary_structure(2,2)
        pred, num_labels = label(pred, structure)
        self.isMobile = np.zeros((self.data.shape[0]), dtype=int)
        for i, point in enumerate(self.data):
            x_img = int(point[0]/0.16 + 512)
            y_img = int(point[1]/0.16 + 512)
            if x_img>=0 and x_img<1024 and y_img>=0 and y_img<1024:
                self.isMobile[i] = pred[x_img, y_img]

    def apply_transformation(self, matrix):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(self.data)
        pc.transform(matrix)
        self.data = np.asarray(pc.points)

    def _compute_cluster_centroids(self, visualize=False):
        if hasattr(self, 'isMobile'):
            cluster_ids = list(np.unique(self.isMobile))
            cluster_ids.remove(0)
            self.centroids = np.zeros((len(cluster_ids),3))
            for c_index, c_id in enumerate(cluster_ids):
                cluster_points = self.data[self.isMobile==c_id,:]
                cluster_center = np.mean(cluster_points, axis=0)
                self.centroids[c_index, :] = cluster_center
            if visualize:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(self.data)

                palette = sns.color_palette("hsv", n_colors=150)
                get_color = lambda tag:palette[tag%150] if tag != 0 else (1.0,1.0,1.0)
                colors = np.array(np.vectorize(get_color)(self.isMobile)).T
                pcd.colors = o3d.utility.Vector3dVector(colors)
                
                pcd_centroids = o3d.geometry.PointCloud()
                pcd_centroids.points = o3d.utility.Vector3dVector(self.centroids)

                pcd_centroids.paint_uniform_color([0, 0, 0])
                o3d.visualization.draw_geometries([pcd, pcd_centroids])

        else:
            raise Exception("First classify the points")

    def compare_using_classifier(self, target, classifier=None, threshold=.5, voxel_size=0.25, initial_transformation=None, initial_centroids=None):

        if not hasattr(self, 'isMobile'):
            self.classify_mobile(classifier, threshold=threshold)
            print("qui")
            if not hasattr(self, 'centroids'):
                print("quo")
                self._compute_cluster_centroids()
        if not hasattr(target, 'isMobile'):
            print("qua")
            target.classify_mobile(classifier, threshold=threshold)
            if not hasattr(target, 'centroids'):
                print("qua")
                target._compute_cluster_centroids()

        background_a = self.data[self.isMobile == 0, :]
        background_b = target.data[target.isMobile == 0, :]
        background_a = background_a[background_a[:,2] > -0.9]
        background_b = background_b[background_b[:,2] > -0.9]
        transformation = self.icp_register(background_a, background_b, ignore_center=True, init=initial_transformation)
        pc_a = o3d.geometry.PointCloud()
        pc_a.points = o3d.utility.Vector3dVector(background_a)
        pc_a.transform(transformation)
        background_a = np.asarray(pc_a.points)
        min_x = min(np.min(background_a[:, 0]),np.min(background_b[:, 0]))
        min_y = min(np.min(background_a[:, 1]),np.min(background_b[:, 1]))
        min_z = min(np.min(background_a[:, 2]),np.min(background_b[:, 2]))
        max_x = max(np.max(background_a[:, 0]),np.max(background_b[:, 0]))
        max_y = max(np.max(background_a[:, 1]),np.max(background_b[:, 1]))
        max_z = max(np.max(background_a[:, 2]),np.max(background_b[:, 2]))
        boundaries = np.array([[min_x, max_x], [min_y, max_y], [min_z, max_z]])
        vox_back_a = SelmaPointCloud(background_a).voxelize(voxel_size, boundaries=boundaries)
        vox_back_b = SelmaPointCloud(background_b).voxelize(voxel_size, boundaries=boundaries)

        if initial_centroids is None:
            initial_centroids = self.centroids
        pc_a = o3d.geometry.PointCloud()
        pc_a.points = o3d.utility.Vector3dVector(self.centroids)
        pc_a.transform(transformation)
        original_centroids = np.asarray(pc_a.points)
        distances = cdist(initial_centroids, target.centroids)
        closest_indices = np.argmin(distances, axis=1)
        new_cluster_positions = target.centroids[closest_indices, :]
        # mobile_a = self.data[self.isMobile == 1, :]
        # mobile_b = target.data[target.isMobile == 1, :]
        # pc_a = o3d.geometry.PointCloud()
        # pc_a.points = o3d.utility.Vector3dVector(mobile_a)
        # pc_a.transform(transformation)
        # mobile_a = np.asarray(pc_a.points)
        # min_x = min(np.min(mobile_a[:, 0]),np.min(mobile_b[:, 0]))
        # min_y = min(np.min(mobile_a[:, 1]),np.min(mobile_b[:, 1]))
        # min_z = min(np.min(mobile_a[:, 2]),np.min(mobile_b[:, 2]))
        # max_x = max(np.max(mobile_a[:, 0]),np.max(mobile_b[:, 0]))
        # max_y = max(np.max(mobile_a[:, 1]),np.max(mobile_b[:, 1]))
        # max_z = max(np.max(mobile_a[:, 2]),np.max(mobile_b[:, 2]))
        # boundaries = np.array([[min_x, max_x], [min_y, max_y], [min_z, max_z]])
        # vox_mobile_a = SelmaPointCloud(mobile_a)
        # vox_mobile_b = SelmaPointCloud(mobile_b)
        # normalized_distance_bw_clusters = np.sum(np.sqrt(np.sum((new_cluster_positions - original_centroids)**2, axis=1)))
        dist_wrt_prev = np.sqrt(np.sum((new_cluster_positions - initial_centroids)**2, axis=1))
        dist_wrt_original = np.sqrt(np.sum((new_cluster_positions - original_centroids)**2, axis=1))
        new_cluster_positions[dist_wrt_prev>40/30] = initial_centroids[dist_wrt_prev>40/30]
        
        normalized_distance_bw_clusters = np.mean(dist_wrt_original[dist_wrt_prev<40/30])

        return vox_back_a.compute_intersection_size(vox_back_b), normalized_distance_bw_clusters, transformation, new_cluster_positions

    def really_euristic_classifier(self, floor_level=-1.26, max_height=2.2, max_width=8, voxel_size=1.25, visualize=False, return_stats=False):
        mask_above_ground = self.data[:,2] > floor_level
        mask_too_high = self.data[:,2] > max_height + floor_level
        human_level = self.data[np.logical_and(mask_above_ground, np.logical_not(mask_too_high)), :]

        min_x = np.min(human_level[:, 0])
        min_y = np.min(human_level[:, 1])
        max_x = np.max(human_level[:, 0])
        max_y = np.max(human_level[:, 1])

        boundaries = np.array([[min_x, max_x], [min_y, max_y]])

        above_ground = self.data[mask_above_ground, :]

        x = np.floor((above_ground[:, 0] - boundaries[0,0]) / voxel_size).astype(int)
        y = np.floor((above_ground[:, 1] - boundaries[1,0]) / voxel_size).astype(int)

        shape = (int(np.ceil((boundaries[0][1]-boundaries[0][0])/voxel_size)), 
                 int(np.ceil((boundaries[1][1]-boundaries[1][0])/voxel_size)))
        
        point_mask = (x < shape[0]) & (y < shape[1])

        x = x[point_mask]
        y = y[point_mask]

        flatten = np.zeros(shape,dtype=int)
        flatten[x, y] = 1
        clusters , labels = label(flatten, generate_binary_structure(2, 2))

        c_id = np.unique(clusters)
        for id in zip(c_id):
            ex, why = np.where(clusters == id)
            if max(ex)-min(ex) > max_width or max(why)-min(why)> max_width:
                clusters[clusters==id]=0


        too_high = self.data[mask_too_high, :]

        x = np.floor((too_high[:, 0] - boundaries[0,0]) / voxel_size).astype(int)
        y = np.floor((too_high[:, 1] - boundaries[1,0]) / voxel_size).astype(int)

        point_mask = (x < shape[0]) & (y < shape[1])

        x = x[point_mask]
        y = y[point_mask]


        clusters[x,y] = 0
        
        all_x = np.floor((self.data[:, 0] - boundaries[0,0]) / voxel_size).astype(int)
        all_y = np.floor((self.data[:, 1] - boundaries[1,0]) / voxel_size).astype(int)
        self.isMobile = np.zeros((self.data.shape[0]), dtype=int)

        for i, id in enumerate(np.unique(clusters)):
            if id != 0:
                cluster_x, cluster_y = np.where(clusters == id)
                for x, y in zip(cluster_x, cluster_y):
                    self.isMobile[(all_x == x) & (all_y==y) & (self.data[:,2] > floor_level)] = i

        self._compute_cluster_centroids(visualize=visualize)
        if return_stats:
            true_mobile = (self.ground_truth[:,0] == 12) | (self.ground_truth[:,0] == 13) | (self.ground_truth[:,0] == 14)
            inferred_mobile = self.isMobile != 0
            return confusion_matrix(true_mobile, inferred_mobile)

    def _compute_chamfer_distance(self, source:o3d.geometry.PointCloud, target:o3d.geometry.PointCloud):
        d1 = source.compute_point_cloud_distance(target)
        d2 = target.compute_point_cloud_distance(source)
        return np.sum(np.asarray(d1)**2) + np.sum(np.asarray(d2)**2)

    def align_and_compute_cd(self, target, mode=NOTHING, floor_level=-1.26, init_transform=None, remove_far_points=POST_REGISTRATION, max_distance=50):
        # Removing floor
        sample_a = self.data[self.data[:,2] > floor_level]
        sample_b = target.data[target.data[:,2] > floor_level]

        if remove_far_points == PRE_REGISTRATION:
            sample_a = sample_a[np.sum(sample_a[:,[0,1]]**2, axis=1) < max_distance**2]
            sample_b = sample_b[np.sum(sample_b[:,[0,1]]**2, axis=1) < max_distance**2]
        # Alignin or maybe not
        if mode == ICP_REGISTRATION:
            transformation = self.icp_register(sample_a, sample_b, init=init_transform)
            if remove_far_points == POST_REGISTRATION:
                sample_a = sample_a[np.sum(sample_a[:,[0,1]]**2, axis=1) < max_distance**2]
                sample_b = sample_b[np.sum(sample_b[:,[0,1]]**2, axis=1) < max_distance**2]
            pc_a = o3d.geometry.PointCloud()
            pc_a.points = o3d.utility.Vector3dVector(sample_a)
            pc_a.transform(transformation)
            pc_b = o3d.geometry.PointCloud()
            pc_b.points = o3d.utility.Vector3dVector(sample_b)
        elif mode == NOTHING:
            pc_a = o3d.geometry.PointCloud()
            pc_a.points = o3d.utility.Vector3dVector(sample_a)
            pc_b = o3d.geometry.PointCloud()
            pc_b.points = o3d.utility.Vector3dVector(sample_b)
        if mode == ICP_REGISTRATION:
            return self._compute_chamfer_distance(pc_a, pc_b), transformation
        else:
            return self._compute_chamfer_distance(pc_a, pc_b)
