import numpy as np


def voxelize(sample: dict, voxel_dimension: float, boundaries: list = None):
    if sample["type"] != "PointCloud":
        raise ValueError("Incopatible sample type")
    if boundaries is None:

        min_x = np.min(sample["data"][:, 0])
        min_y = np.min(sample["data"][:, 1])
        min_z = np.min(sample["data"][:, 2])
        max_x = np.max(sample["data"][:, 0])
        max_y = np.max(sample["data"][:, 1])
        max_z = np.max(sample["data"][:, 2])

        boundaries = np.array([[min_x, max_x], [min_y, max_y], [min_z, max_z]])

    # Create the matrix for the voxelization
    voxels = np.zeros((int(np.ceil((boundaries[0][1]-boundaries[0][0])/voxel_dimension)), 
                       int(np.ceil((boundaries[1][1]-boundaries[1][0])/voxel_dimension)), 
                       int(np.ceil((boundaries[2][1]-boundaries[2][0])/voxel_dimension))), dtype=int)
    
    for point in sample["data"]:
        x = int(np.floor((point[0]-boundaries[0,0])/voxel_dimension))
        y = int(np.floor((point[1]-boundaries[1,0])/voxel_dimension))
        z = int(np.floor((point[2]-boundaries[2,0])/voxel_dimension))
        if x < voxels.shape[0] and y < voxels.shape[1] and z < voxels.shape[2]:
            voxels[x, y, z] = 1 #TODO try += 1
        
    return {
        "type":"Voxels",
        "boundaries":boundaries,
        "data":voxels,
        "voxel_dimension":voxel_dimension
    }

def convert_voxels_to_pointcloud(sample:dict):
    if sample["type"] != "Voxels":
        raise ValueError("Incopatible sample type")
    points = np.array(np.where(sample["data"])).T*sample["voxel_dimension"] + sample["boundaries"][:,0]
    return {
        "type":"PointCloud",
        "data":points
    }
