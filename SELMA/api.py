import os
from enum import Enum
from plyfile import PlyData
import numpy as np
import h5py

class Town(Enum):
    """Town available in the dataset

    Args:
        Enum (str): name of the town
    """
    T1 = "Town01"
    T2 = "Town02"
    T3 = "Town03"
    T4 = "Town04"
    T5 = "Town05"
    T6 = "Town06"
    T7 = "Town07"
    T10 = "Town10HD"


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


class Time(Enum):
    """Time available in the dataset

    Args:
        Enum (str): time moment
    """
    Night = "Night"
    Noon = "Noon"
    Sunset = "Sunset"


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

    def getEstention(self):
        """get the estention of the stored file according to the sensor

        Returns:
            str: estention of the file representing those objects
        """
        if self.value.find("CAM_") == -1:
            return ".ply"
        else:
            return ".jpg"


class Dataset:

    def __init__(self, root: str):
        """Initializes the dataset

        Args:
            root (str): path to the root
        """
        self.root_path = root

    def _get_path_folder(self, scope: str, town: Town, weather: Weather, time: Time, sensor: Sensor):
        """Get the folder where the specified data are stored

        Args:
            scope (str): scope of the dataset can be CV or ...
            town (Town): The town in which the simulaiton has been performed
            weather (Weather): The wether with which the simulaiton has been performed
            time (Time): The time in which the simulaiton has been performed
            sensor (Sensor): The sensor mesuring in the simulaiton

        Returns:
            str: path to the desired folder
        """
        return os.path.join(self.root_path,
                            scope,
                            "dataset",
                            town.value+"_Opt_"+weather.value+time.value,
                            sensor.value)

    def _get_path_file(self, scope: str, town: Town, weather: Weather, time: Time, sensor: Sensor, id: int):
        """Get the path where the desired data are stored

        Args:
            scope (str): scope of the dataset can be CV or ...
            town (Town): The town in which the simulaiton has been performed
            weather (Weather): The wether with which the simulaiton has been performed
            time (Time): The time in which the simulaiton has been performed
            sensor (Sensor): The sensor mesuring in the simulaiton
            id (int): the ID of the sample

        Returns:
            str: path to the desired sample
        """
        return os.path.join(self._get_path_folder(scope, town, weather, time, sensor),
                            town.value+"_Opt_"+weather.value+time.value+"_"+str(id)+sensor.getEstention())

    def get_ids(self, scope: str, town: Town, weather: Weather, time: Time, sensor: Sensor):
        """get all the IDs of a specific simulation

        Args:
            scope (str): scope of the dataset can be CV or ...
            town (Town): The town in which the simulaiton has been performed
            weather (Weather): The wether with which the simulaiton has been performed
            time (Time): The time in which the simulaiton has been performed
            sensor (Sensor): The sensor mesuring in the simulaiton

        Returns:
            list(int): list of all the available samples
        """
        folder = self._get_path_folder(scope, town, weather, time, sensor)
        folder_content = os.listdir(folder)
        folder_content = [file for file in folder_content if file.endswith(sensor.getEstention())]
        ids = [int(file.split('_')[3][:-4]) for file in folder_content]
        return ids
    
    def open_sample(self, scope: str, town: Town, weather: Weather, time: Time, sensor: Sensor, id:int):
        """returns the value of the selected sample

        Args:
            scope (str): scope of the dataset can be CV or ...
            town (Town): The town in which the simulaiton has been performed
            weather (Weather): The wether with which the simulaiton has been performed
            time (Time): The time in which the simulaiton has been performed
            sensor (Sensor): The sensor mesuring in the simulaiton
            id (int): the ID of the sample

        Raises:
            ValueError: if the function does not know how to handle the format

        Returns:
            dict: with keys ["type", "data", "class"] containing respectively the type of the data, the data itself and the ground truth for each sample in data
        """
        file_path = self._get_path_file(scope, town, weather, time, sensor, id)
        out = {}
        if (sensor.getEstention() == ".ply"):
            out["type"] = "PointCloud"
            data = PlyData.read(file_path)
            points = [data['vertex'][axis] for axis in ['x', 'y', 'z']]
            points = np.array(points).T
            objTag = data['vertex']['ObjTag']
            out["data"] = points
            out["class"] = objTag
        elif (sensor.getEstention() == ".jpg"):
            raise ValueError("Implementami")
        else:
            raise ValueError("Unkown file estention")
        return out

        
    