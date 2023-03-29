import h5py
import numpy as np
import open3d as o3d
from . import selmaPointCloud
from . import datasetApi
from tqdm import trange
import seaborn as sns

def create_homogeneous_matrix(x, y, z, pitch, yaw, roll):
    # Convert pitch, yaw, and roll angles to radians
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)
    roll = np.radians(roll)

    # Create rotation matrices for each angle
    Rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    Rz = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])

    # Combine the rotation matrices
    R = np.dot(Rz, np.dot(Ry, Rx))

    # Create the translation vector
    t = np.array([[x], [y], [z]])

    # Combine the rotation matrix and translation vector into a homogeneous matrix
    T = np.hstack((R, t))
    T = np.vstack((T, np.array([0, 0, 0, 1])))

    return T

def find_truth(path_to_bbox, starting_index, number_of_samples, dataset, id, weather, time, sensor):
    VOXEL_SIZE = 1

    diffs = []
    with h5py.File(path_to_bbox,'r') as f:
        root_grp = f.get("BBOX")
        ids = list(root_grp.keys())
        for id_actor in ids:
            agent = root_grp.get(id_actor)
            if "vehicle" in agent.attrs['type']:
                ego = root_grp.get(id_actor)
                break
        loc = np.array(ego.get('location'))
        rot = np.array(ego.get('rotation'))
    first_pointcloud = dataset.open_measurement_sample_TLC(id, weather, time, sensor, starting_index)
    first_environment = None
    first_tags = None
    first_mobile =  None
    first = True
    for i in trange(starting_index, starting_index+number_of_samples):
        actual_diff = {"new" : 0}
        # m = create_homogeneous_matrix(-loc[i-1,1], -loc[i-1,0], -loc[i-1,2], 180-rot[i-1,0], 180-rot[i-1,2], 180-rot[i-1,1])
        m = create_homogeneous_matrix(-loc[i-1,0], -loc[i-1,1], -loc[i-1,2], 180-rot[i-1,0], 180-rot[i-1,1], 180-rot[i-1,2])
        lt = dataset.open_measurement_sample_TLC(id, weather, time, sensor, i)
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(lt.data)
        pc.transform(m)
        tags = np.unique(lt.ground_truth[:,1])
        if first:
            first_pointcloud.data = np.asarray(pc.points)
            first_tags = tags
            first_mobile = selmaPointCloud.SelmaPointCloud(first_pointcloud.data[first_pointcloud.ground_truth[:,1] !=0,:])

        actual_mobile = selmaPointCloud.SelmaPointCloud(np.asarray(pc.points)[lt.ground_truth[:,1] !=0,:])
        actual_diff["mobile"] = first_mobile.intersection_using_voxels(actual_mobile, VOXEL_SIZE, crop_street=True)

        for tag in tags:
            temp_points = np.asarray(pc.points)
            temp_points = temp_points[lt.ground_truth[:,1] == tag]

            if tag == 0:
                if first:
                    first = False
                    first_environment = selmaPointCloud.SelmaPointCloud(temp_points)
                    actual_diff["background"] = 0
                else:
                    actual_environment = selmaPointCloud.SelmaPointCloud(temp_points)
                    actual_diff["background"] = first_environment.intersection_using_voxels(actual_environment, VOXEL_SIZE, crop_street=True)
            else:
                if tag in first_tags:
                    actual_diff[tag] = np.sqrt(np.sum((np.mean(first_pointcloud.data[first_pointcloud.ground_truth[:,1] == tag], axis=0) - np.mean(temp_points,axis=0))**2))
                else:
                    actual_diff["new"] += 1
        for t in set(first_tags) - set(tags):
            actual_diff[t] = 0
        diffs.append(actual_diff)
    return diffs

def find_truth_cd(path_to_bbox, starting_index, number_of_samples, dataset, id, weather, time, sensor, max_distance=50, visualize=False):
    pal = sns.color_palette("rocket", number_of_samples)
    j=0
    pcs = []
    cds = []
    with h5py.File(path_to_bbox,'r') as f:
        root_grp = f.get("BBOX")
        ids = list(root_grp.keys())
        for id_actor in ids:
            agent = root_grp.get(id_actor)
            if "vehicle" in agent.attrs['type']:
                ego = root_grp.get(id_actor)
                break
        loc = np.array(ego.get('location'))
        rot = np.array(ego.get('rotation'))

    first_pointcloud = dataset.open_measurement_sample_TLC(id, weather, time, sensor, starting_index)
    first_pc = o3d.geometry.PointCloud()
    dd = first_pointcloud.data
    dd = dd[np.sum(dd[:,[0,1]]**2, axis=1) < max_distance**2]
    dd = dd[dd[:,2]>-1.26]
    first_pc.points = o3d.utility.Vector3dVector(dd)
    m = create_homogeneous_matrix(-loc[starting_index-1,1], -loc[starting_index-1,0], -loc[starting_index-1,2], 180-rot[starting_index-1,0], 180-rot[starting_index-1,2], 180-rot[starting_index-1,1])
    first_pc.transform(m)
    if visualize:
        first_pc.paint_uniform_color(pal[j])
        j+=1
    pcs.append(first_pc)
    
    for i in trange(starting_index + 1, starting_index+number_of_samples):
        if visualize:
            i = i*10
        m = create_homogeneous_matrix(-loc[i-1,1], -loc[i-1,0], -loc[i-1,2], 180-rot[i-1,0], 180-rot[i-1,2], 180-rot[i-1,1])
        lt = dataset.open_measurement_sample_TLC(id, weather, time, sensor, i)
        other_dd = lt.data
        other_dd = other_dd[np.sum(other_dd[:,[0,1]]**2, axis=1) < max_distance**2]
        other_dd = other_dd[other_dd[:,2]>-1.26]
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(other_dd)
        pc.transform(m)
        if visualize:
            pc.paint_uniform_color(pal[j])
            j+=1
        pcs.append(pc)
        d1 = first_pc.compute_point_cloud_distance(pc)
        d2 = pc.compute_point_cloud_distance(first_pc)
        cd = np.sum(np.asarray(d1)**2) + np.sum(np.asarray(d2)**2)
        cds.append(cd)
    if visualize:
        o3d.visualization.draw_geometries(pcs)

    return cds

def find_truth_cd_matching_ball(path_to_bbox, starting_index, number_of_samples, dataset, id, weather, time, sensor, max_distance=50, visualize=False):
    pal = sns.color_palette("rocket", number_of_samples)
    j=0
    pcs = []
    cds = []
    with h5py.File(path_to_bbox,'r') as f:
        root_grp = f.get("BBOX")
        ids = list(root_grp.keys())
        for id_actor in ids:
            agent = root_grp.get(id_actor)
            if "vehicle" in agent.attrs['type']:
                ego = root_grp.get(id_actor)
                break
        loc = np.array(ego.get('location'))
        rot = np.array(ego.get('rotation'))

    first_pointcloud = dataset.open_measurement_sample_TLC(id, weather, time, sensor, starting_index)
    first_pc = o3d.geometry.PointCloud()
    dd = first_pointcloud.data
    # dd = dd[dd[:,2]>-1.26]
    first_pc.points = o3d.utility.Vector3dVector(dd)
    m = create_homogeneous_matrix(-loc[starting_index-1,1], -loc[starting_index-1,0], -loc[starting_index-1,2], 180-rot[starting_index-1,0], 180-rot[starting_index-1,2], 180-rot[starting_index-1,1])
    first_pc.transform(m)
    if visualize:
        first_pc.paint_uniform_color(pal[j])
        j+=1
    pcs.append(first_pc)
    first_data = np.asarray(first_pc.points)
    
    for i in trange(starting_index + 1, starting_index+number_of_samples):
        if visualize:
            i = i*100
        m = create_homogeneous_matrix(-loc[i-1,1], -loc[i-1,0], -loc[i-1,2], 180-rot[i-1,0], 180-rot[i-1,2], 180-rot[i-1,1])
        lt = dataset.open_measurement_sample_TLC(id, weather, time, sensor, i)
        other_dd = lt.data
        other_dd = other_dd[np.sum(other_dd[:,[0,1]]**2, axis=1) < max_distance**2]
        # other_dd = other_dd[other_dd[:,2]>-1.26]
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(other_dd)
        pc.transform(m)
        first_ball_mask = np.sum((first_data[:, [0,1]] + [loc[i-1,1], loc[i-1,0]])**2, axis=1) < max_distance**2
        first_ball_data = first_data[first_ball_mask]
        first_ball_pc = o3d.geometry.PointCloud()
        first_ball_pc.points = o3d.utility.Vector3dVector(first_ball_data)
        if visualize:
            pc.paint_uniform_color(pal[j])
            first_ball_pc.paint_uniform_color(pal[0])
            # o3d.visualization.draw_geometries([first_ball_pc, pc])
            j+=1
        pcs.append(pc)
        d1 = first_ball_pc.compute_point_cloud_distance(pc)
        d2 = pc.compute_point_cloud_distance(first_ball_pc)
        cd = np.sum(np.asarray(d1)**2) + np.sum(np.asarray(d2)**2)
        cds.append(cd)
    if visualize:
        o3d.visualization.draw_geometries(pcs)

    return cds

def find_next_transmission(path_to_bbox, starting_index, dataset, id, weather, time, sensor, max_distance=20, max_cd = 1000):
    cds = []
    with h5py.File(path_to_bbox,'r') as f:
        root_grp = f.get("BBOX")
        ids = list(root_grp.keys())
        for id_actor in ids:
            agent = root_grp.get(id_actor)
            if "vehicle" in agent.attrs['type']:
                ego = root_grp.get(id_actor)
                break
        loc = np.array(ego.get('location'))
        rot = np.array(ego.get('rotation'))

    first_pointcloud = dataset.open_measurement_sample_TLC(id, weather, time, sensor, starting_index)
    first_pc = o3d.geometry.PointCloud()
    dd = first_pointcloud.data
    dd = dd[dd[:,2]>-1.26]
    first_pc.points = o3d.utility.Vector3dVector(dd)
    m = create_homogeneous_matrix(-loc[starting_index-1,1], -loc[starting_index-1,0], -loc[starting_index-1,2], 180-rot[starting_index-1,0], 180-rot[starting_index-1,2], 180-rot[starting_index-1,1])
    first_pc.transform(m)
    first_data = np.asarray(first_pc.points)
    cd = 0
    i = starting_index
    sim_length = dataset.get_measurement_series_length_TLC(id, weather, time, sensor)
    while cd < max_cd and i < sim_length-1:
        i += 1
        m = create_homogeneous_matrix(-loc[i-1,1], -loc[i-1,0], -loc[i-1,2], 180-rot[i-1,0], 180-rot[i-1,2], 180-rot[i-1,1])
        lt = dataset.open_measurement_sample_TLC(id, weather, time, sensor, i)
        other_dd = lt.data
        other_dd = other_dd[np.sum(other_dd[:,[0,1]]**2, axis=1) < max_distance**2]
        other_dd = other_dd[other_dd[:,2]>-1.26]
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(other_dd)
        pc.transform(m)
        first_ball_mask = np.sum((first_data[:, [0,1]] + [loc[i-1,1], loc[i-1,0]])**2, axis=1) < max_distance**2
        first_ball_data = first_data[first_ball_mask]
        first_ball_pc = o3d.geometry.PointCloud()
        first_ball_pc.points = o3d.utility.Vector3dVector(first_ball_data)
        d1 = first_ball_pc.compute_point_cloud_distance(pc)
        d2 = pc.compute_point_cloud_distance(first_ball_pc)
        cd = np.sum(np.asarray(d1)**2) + np.sum(np.asarray(d2)**2)
        cds.append(cd)
    return cds, i

def plot_all_toghether(dataset:datasetApi.Dataset, t, t_max, route, weather, time, sensors):
    # def _create_homogeneous_matrix_BIS(x, y, z, x_rotation, y_rotation, z_rotation):
    #     # Convert x_rotation, z_rotation, and y_rotation angles to radians
    #     x_rotation = np.radians(x_rotation)
    #     y_rotation = np.radians(y_rotation)
    #     z_rotation = np.radians(z_rotation)

    #     # Create rotation matrices for each angle
    #     Rx = np.array([[1, 0, 0], [0, np.cos(x_rotation), -np.sin(x_rotation)], [0, np.sin(x_rotation), np.cos(x_rotation)]])
    #     Ry = np.array([[np.cos(y_rotation), 0, np.sin(y_rotation)], [0, 1, 0], [-np.sin(y_rotation), 0, np.cos(y_rotation)]])
    #     Rz = np.array([[np.cos(z_rotation), -np.sin(z_rotation), 0], [np.sin(z_rotation), np.cos(z_rotation), 0], [0, 0, 1]])

    #     # Combine the rotation matrices
    #     R = np.dot(Rz, np.dot(Ry, Rx))

    #     # Create the translation vector
    #     t = np.array([[x], [y], [z]])

    #     # Combine the rotation matrix and translation vector into a homogeneous matrix
    #     T = np.hstack((R, t))
    #     T = np.vstack((T, np.array([0, 0, 0, 1])))
    #     return T

    if t_max == 0:
        number_of_samples = dataset.get_measurement_series_length_TLC(route, weather, time, sensors[0])
    else:
        number_of_samples = t_max
    pals = [sns.color_palette("viridis", int(np.ceil(number_of_samples)/t)+2),
            sns.color_palette("rocket", int(np.ceil(number_of_samples)/t)+2),
            sns.color_palette("plasma", int(np.ceil(number_of_samples)/t)+2)]
    j=0
    pcs = []
    with h5py.File(dataset.get_path_bbox(route, weather, time),'r') as f:
        root_grp = f.get("BBOX")
        ids = list(root_grp.keys())
        for id_actor in ids:
            agent = root_grp.get(id_actor)
            if "vehicle" in agent.attrs['type']:
                ego = root_grp.get(id_actor)
                break
        loc = np.array(ego.get('location'))
        rot = np.array(ego.get('rotation'))
    for i in trange(1, number_of_samples, t):
        for idx_s, s in enumerate(sensors):
            # print(loc[i-1,2])
            # m = _create_homogeneous_matrix_BIS(-loc[i-1,1], -loc[i-1,0], loc[i-1,2], -rot[i-1,2], -rot[i-1,0], -rot[i-1,1])
            # m = _create_homogeneous_matrix_BIS(-loc[i-1,1], -loc[i-1,0], -loc[i-1,2], -rot[i-1,2], -rot[i-1,0], -rot[i-1,1])
            lt = dataset.open_measurement_sample_TLC(route, weather, time, s, i)
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(lt.data)
            # pc.transform(m)
            # print(pc.get_rotation_matrix_from_xyz((np.radians(-rot[i-1,2]), np.radians(-rot[i-1,0]), np.radians(-rot[i-1,1]))))
            # print(m)
            T_sens = s.get_homogeneous_matrix()
            T_rot = np.eye(4)
            T_rot[:3, :3] = pc.get_rotation_matrix_from_xyz((np.radians(-rot[i-1,2]), np.radians(-rot[i-1,0]), np.radians(-rot[i-1,1])))
            T_loc = np.eye(4)
            T_loc[0:3,3] = [-loc[i-1,1], -loc[i-1,0], loc[i-1,2]]
            T = np.dot(T_loc, T_rot)
            T = np.dot(T, T_sens)
            # T[0, 3] = -loc[i-1,1]
            # T[1, 3] = -loc[i-1,0]
            # T[2, 3] = -loc[i-1,2]
            # pc.rotate(pc.get_rotation_matrix_from_xyz((np.radians(-rot[i-1,2]), np.radians(-rot[i-1,0]), np.radians(-rot[i-1,1]))), center=(0,0,0))
            # pc.translate((-loc[i-1,1], -loc[i-1,0], loc[i-1,2]))
            pc.transform(T)
            pc.paint_uniform_color(pals[idx_s][j])
            pcs.append(pc)
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                        #size=100, origin=[-loc[0,1], -loc[0,0], 0])
                        size=5, origin=[-loc[i-1,1], -loc[i-1,0], loc[i-1,2]])
            pcs.append(mesh_frame)
        j+=1
    o3d.visualization.draw_geometries(pcs)
