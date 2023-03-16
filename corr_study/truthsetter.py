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
    for i in range(starting_index, starting_index+number_of_samples):
        actual_diff = {"new" : 0}
        m = create_homogeneous_matrix(-loc[i-1,1], -loc[i-1,0], -loc[i-1,2], 180-rot[i-1,0], 180-rot[i-1,2], 180-rot[i-1,1])
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