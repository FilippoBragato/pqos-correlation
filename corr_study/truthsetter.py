import h5py
import numpy as np
import open3d as o3d
from . import selmaPointCloud

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
    with h5py.File(path_to_bbox,'r') as f:
        root_grp = f.get("BBOX")
        ids = list(root_grp.keys())
        ego = root_grp.get("0194")
        loc = np.array(ego.get('location'))
        rot = np.array(ego.get('rotation'))

    first_pointcloud = dataset.open_measurement_sample_TLC(id, weather, time, sensor, starting_index)
    print(first_pointcloud.data.shape)
    first = True
    for i in range(starting_index, starting_index+number_of_samples):
        m = create_homogeneous_matrix(-loc[i-1,1], -loc[i-1,0], -loc[i-1,2], 180-rot[i-1,0], 180-rot[i-1,2], 180-rot[i-1,1])
        lt = dataset.open_measurement_sample_TLC(id, weather, time, sensor, i)
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(lt.data)
        pc.transform(m)
        if first:
            first = False
            first_pointcloud.data = np.asarray(pc.points)
        tags = np.unique(lt.ground_truth[:,1])
        for tag in tags:
            temp_points = np.asarray(pc.points)
            temp_points = temp_points[lt.ground_truth[:,1] == tag]
            print(first_pointcloud.data.shape)
            print(first_pointcloud.ground_truth.shape)
            print(tag, np.mean(first_pointcloud.data[first_pointcloud.ground_truth[:,1] == tag], axis=0) - np.mean(temp_points,axis=0))