import h5py
import numpy as np
import glob
import json
import pandas as pd
from enum import Enum
from plyfile import PlyData, PlyElement
import open3d as o3d
import os
from tqdm import trange

class Town(Enum):
    """towns available in the dataset
    
    Args:
        Enum (str): description of the town
    """
    T1 = "Town01_Opt"
    T2 = "Town02_Opt"
    T3 = "Town03_Opt"
    T4 = "Town04_Opt"
    T5 = "Town05_Opt"
    T6 = "Town06_Opt"
    T7 = "Town07_Opt"
    T10 = "Town10_OptHD"

class Traffic(Enum):
    """traffic available in the dataset

    Args:
        Enum (str): description of the traffic
    """
    High = "High"
    Low = "Low"
    Medium = "Medium"
    None_ = "None"

class Time(Enum):
    """Time available in the dataset

    Args:
        Enum (str): time moment
    """
    Night = "Night"
    Noon = "Noon"
    Sunset = "Sunset"

class Weather(Enum):
    """weather available in the dataset

    Args:
        Enum (str): description of the weather
    """
    Clear = "Clear"
    Cloudy = "Cloudy"
    HardFog = "HardFog"
    HardRain = "HardRain"
    MidFog = "MidFog"
    MidRain = "MidRain"
    SoftRain = "SoftRain"
    Wet = "Wet"
    WetCloudy = "WetCloudy"

class Sensor(Enum):
    """Sensor available in the dataset

    Args:
        Enum (str): name and position of the sensor
    """
    CB = "CAM_BACK"
    CD = "CAM_DESK"
    CF = "CAM_FRONT"
    CFL = "CAM_FRONT_LEFT"
    CFR = "CAM_FRONT_RIGHT"
    CL = "CAM_LEFT"
    CR = "CAM_RIGHT"
    DB = "DEPTHCAM_BACK"
    DD = "DEPTHCAM_DESK"
    DF = "DEPTHCAM_FRONT"
    DFL = "DEPTHCAM_FRONT_LEFT"
    DFR = "DEPTHCAM_FRONT_RIGHT"
    DL = "DEPTHCAM_LEFT"
    DR = "DEPTHCAM_RIGHT"
    LFL = "LIDAR_FRONT_LEFT"
    LFR = "LIDAR_FRONT_RIGHT"
    LT = "LIDAR_TOP"
    SB = "SEGCAM_BACK"
    SD = "SEGCAM_DESK"
    SF = "SEGCAM_FRONT"
    SFL = "SEGCAM_FRONT_LEFT"
    SFR = "SEGCAM_FRONT_RIGHT"
    SL = "SEGCAM_LEFT"
    SR = "SEGCAM_RIGHT"
    BB = "BBOX"

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        elif isinstance(other, Sensor):
            return self.value == other.value
        else:
            return False

    def get_estention(self):
        """get the estention of the stored file according to the sensor

        Returns:
            str: estention of the file representing those objects
        """
        if self.value.find("CAM_") == -1:
            return ".hdf5"
        else:
            return ".mp4"

def get_filepath(root:str, town:Town, traffic:Traffic, id:int, weather:Weather, time:Time, sensor:Sensor):
    """get the filepath of the data

    Args:
        root (str): root path of the data
        town (Town): town of the data
        weather (Weather): weather of the data
        time (Time): time of the data
        sensor (Sensor): sensor of the data

    Returns:
        str: filepath of the data
    """
    if sensor == Sensor.BB:
        print("BBOX")
        filepath = os.path.join(root, town.value, traffic.value, "BBOX.hdf5")
        return filepath
    est = sensor.get_estention()
    weathertime = weather.value + time.value
    if est == ".hdf5":
        weathertime = "None"

    filepath = os.path.join(root, town.value, traffic.value, str(id), sensor.value, weathertime, "data" + sensor.get_estention())
    return filepath

def open_measurement_sample_TLC(filepath, timestep):
    with h5py.File(filepath, "r") as f:
        main_group_key = list(f.keys())[0]
        time_group_key = str(timestep).zfill(5)
        array_data = f[main_group_key][time_group_key][()]
        out = (array_data[:,[0,1,2]], array_data[:,[3,4]].astype(int))
    return out

def open_sensors(folderpath):
    sensor_files = glob.glob(folderpath + '/*.json')
    sensors = {}
    for sensor_file in sensor_files:
        with open(sensor_file) as f:
            sensors_dict = json.load(f)
            for sensor in sensors_dict:
                sensors[sensor] = sensors_dict[sensor]
    return sensors

def open_bounding_boxes(filepath):
    with h5py.File(filepath, "r") as f:
        root_group = f["BBOX"]
        general_data_path = "/home/filo/SELMA_utils/archives/bbox_const.json"
        with open(general_data_path) as f:
            general_data = json.load(f)
        general_data = pd.DataFrame(general_data)
        general_data["BP_ID"] = general_data["ID"]
        #remove the ID column
        general_data = general_data.drop(columns=["ID"])
        # general_data = general_data[["ID", "class"]]

        actor_ids = root_group.keys()
        dfs = []
        for actor_id in actor_ids:
            if actor_id == "Actor Data" or actor_id == "Transforms":
                continue
            actor_group = root_group[actor_id]
            loc_data = actor_group["location"][()]
            rot_data = actor_group["rotation"][()]

            actor_data = pd.DataFrame(loc_data, columns=["location x", "location y", "location z"]) 
            actor_data["rotation x"] = rot_data[:,0]
            actor_data["rotation y"] = rot_data[:,1]
            actor_data["rotation z"] = rot_data[:,2]
            actor_data["BP_ID"] =actor_group.attrs["type"]

            actor_data["ID"] = actor_id
            actor_data["ID"] = actor_data["ID"].astype(int)
            actor_data["time step"] = np.arange(actor_data.shape[0], dtype=int)
            dfs.append(actor_data)
        actor_data = pd.concat(dfs)
        general_data = general_data.merge(actor_data, on="BP_ID", how="left")
    return general_data

def write_to_ply(points, labels, path):
    # create an appropriate array
    points_ply = points.copy()

    points_ply = points_ply*1000 #TUNABLE
    points_ply = points_ply.astype(np.int32)

    red_ply = labels[:,1]#.astype(np.uint8)
    green_ply = (labels[:,0]//256)#.astype(np.uint8)
    blue_ply = (labels[:,0]%256)#.astype(np.uint8)

    colors_ply = np.array([red_ply, green_ply, blue_ply], dtype=np.uint8).T

    pts = list(zip(points_ply[:,0],
                   points_ply[:,1],
                   points_ply[:,2],
                   red_ply,
                   green_ply,
                   blue_ply))

    ply_out = np.array(pts,
                       dtype=[("x", np.int32),
                              ("y", np.int32),
                              ("z", np.int32),
                              ("red", np.uint8),
                              ("green", np.uint8),
                              ("blue", np.uint8)])
    
    el = PlyElement.describe(ply_out, "vertex")
    PlyData([el], byte_order="<").write(path)

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points_ply)
    # pcd.colors = o3d.utility.Vector3dVector(colors_ply)
    # o3d.io.write_point_cloud("test.ply", pcd)

def get_ego_id(bboxpath):
    with h5py.File(bboxpath, "r") as f:
        root_group = f["BBOX"]
        transforms_group = root_group["Transforms"]
        actor_ids = list(transforms_group.keys())
        actor_ids.sort()
        for actor_id in actor_ids:
            if actor_id == "Actor Data" or actor_id == "Transforms":
                continue
            actor_group = transforms_group[actor_id]
            if "vehicle" in actor_group.attrs["type"]:
                return int(actor_id)
    return None

def get_tlc_length(root:str, town:Town, traffic:Traffic, id:int, weather:Weather, time:Time, sensor:Sensor):
    path = get_filepath(root, town, traffic, id, weather, time, sensor)
    with h5py.File(path, "r") as f:
        main_group_key = list(f.keys())[0]
        idxs = list(f[main_group_key].keys())
        return len(idxs)

def tlc_series_to_ply(root:str, town:Town, traffic:Traffic, id:int, weather:Weather, time:Time, sensor:Sensor):
    path = get_filepath(root, town, traffic, id, weather, time, sensor)
    ply_dir = os.path.sep.join(path.split(os.path.sep)[:-1] + ["ply"])
    os.makedirs(ply_dir, exist_ok=True)
    series_len = get_tlc_length(root, town, traffic, id, weather, time, sensor)
    for i in trange(series_len):
        path_out = ply_dir + os.path.sep + str(i).zfill(5) + ".ply"
        pts, gt = open_measurement_sample_TLC(path, i)
        write_to_ply(pts, gt, path_out)

def get_ply_paths(root:str, town:Town, traffic:Traffic, id:int, weather:Weather, time:Time, sensor:Sensor):
    path = get_filepath(root, town, traffic, id, weather, time, sensor)
    ply_dir = os.path.sep.join(path.split(os.path.sep)[:-1] + ["ply"])
    ply_paths = glob.glob(ply_dir + os.path.sep + "*.ply")
    ply_paths.sort()
    return ply_paths

if __name__ == "__main__":

    root = "archives"
    town = Town.T1
    traffic = Traffic.None_
    id = 1
    weather = Weather.Clear
    time = Time.Noon
    sensor = Sensor.LT

    tlc_series_to_ply(root, town, traffic, id, weather, time, sensor)
        
def get_matrix(loc, rot):
    """
    Creates matrix from carla transform.
    """

    c_y = np.cos(np.radians(rot[2]))
    s_y = np.sin(np.radians(rot[2]))
    c_r = np.cos(np.radians(rot[0]))
    s_r = np.sin(np.radians(rot[0]))
    c_p = np.cos(np.radians(rot[1]))
    s_p = np.sin(np.radians(rot[1]))
    matrix = np.matrix(np.identity(4))
    matrix[0, 3] = loc[0]
    matrix[1, 3] = loc[1]
    matrix[2, 3] = loc[2]
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    return matrix