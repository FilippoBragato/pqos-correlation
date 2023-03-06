import os
from enum import Enum
from plyfile import PlyData
import numpy as np
import h5py
from .selmaImage import SelmaImage
from .selmaPointCloud import SelmaPointCloud

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
        
    def getType(self):
        """get the type of the stored file according to the sensor

        Returns:
            str: estention of the file representing those objects
        """
        if self.value.find("CAM_") == -1:
            return "PointCloud"
        else:
            return "Image"


class Dataset:

    def __init__(self, root: str):
        """Initializes the dataset

        Args:
            root (str): path to the root
        """
        self.root_path = root

    def _get_path_archives_TLC(self) -> str:
        """Get the path to the archive folder

        Returns:
            str: path to the archive folder
        """
        return os.path.join(self.root_path, "TLC_web", "archives")
    
    def get_routes_TLC(self):
        """Get all the routes available in the archive folder

        Returns:
            list(str): list of the identifiers of all available folders
        """
        return os.listdir(self._get_path_archives_TLC())
    
    def _get_path_h5(self, route:str, weather:Weather, time:Time, sensor:Sensor) -> str:
        """Get the path to the archive file whit the desired parameters

        Args:
            route (str): identifier of the route
            weather (Weather): The wether with which the simulaiton has been performed
            time (Time): The time in which the simulaiton has been performed
            sensor (Sensor): The sensor mesuring in the simulaiton

        Returns:
            str: path to the archive file
        """
        return os.path.join(self._get_path_archives_TLC(),
                            route,
                            weather.value+time.value,
                            sensor.value,
                            route+"-"+weather.value+time.value+"-"+sensor.value+".hdf5")
    
    def open_measurement_series_TLC(self, route:str, weather:Weather, time:Time, sensor:Sensor):
        """Open the archive of the desired simulation and returns a list containing temporal ordered data

        Args:
            route (str): identifier of the route
            weather (Weather): The wether with which the simulaiton has been performed
            time (Time): The time in which the simulaiton has been performed
            sensor (Sensor): The sensor mesuring in the simulaiton

        Returns:
            list(Obj): list of required data
        """
        out = []
        with h5py.File(self._get_path_h5(route, weather, time, sensor), "r") as f:
            main_group_key = list(f.keys())[0]
            time_group_key = list(f[main_group_key].keys())
            for k in time_group_key:
                if sensor.getType() == "PointCloud":
                    out.append(SelmaPointCloud(f[main_group_key][k][()]), time_step=int(k))
                elif sensor.getType() == "Image":
                    out.append(SelmaImage(f[main_group_key][k][()]), time_step=int(k))
        return out
    
    def open_measurement_sample_TLC(self, route:str, weather:Weather, time:Time, sensor:Sensor, time_step:int):
        """Open the archive of the desired simulation and returns the required sample

        Args:
            route (str): identifier of the route
            weather (Weather): The wether with which the simulaiton has been performed
            time (Time): The time in which the simulaiton has been performed
            sensor (Sensor): The sensor mesuring in the simulaiton
            time_step (int): the time step that has to be extracted from the serie

        Returns:
            obj: the desired item
        """
        with h5py.File(self._get_path_h5(route, weather, time, sensor), "r") as f:
            main_group_key = list(f.keys())[0]
            time_group_key = str(time_step).zfill(5)
            if sensor.getType() == "PointCloud":
                array_data = f[main_group_key][time_group_key][()]
                out = SelmaPointCloud(array_data[:,[0,1,2]], ground_truth=array_data[:,4].astype(int), time_step=time_step)
            elif sensor.getType() == "Image":
                out = SelmaImage(f[main_group_key][time_group_key][()], time_step=time_step)
        return out

    def _get_path_folder_CV(self, town: Town, weather: Weather, time: Time, sensor: Sensor) -> str:
        """Get the folder where the specified data are stored

        Args:
            town (Town): The town in which the simulaiton has been performed
            weather (Weather): The wether with which the simulaiton has been performed
            time (Time): The time in which the simulaiton has been performed
            sensor (Sensor): The sensor mesuring in the simulaiton

        Returns:
            str: path to the desired folder
        """
        return os.path.join(self.root_path,
                            "CV",
                            "dataset",
                            town.value+"_Opt_"+weather.value+time.value,
                            sensor.value)

    def _get_path_file_CV(self, town: Town, weather: Weather, time: Time, sensor: Sensor, id: int) -> str:
        """Get the path where the desired data are stored

        Args:
            town (Town): The town in which the simulaiton has been performed
            weather (Weather): The wether with which the simulaiton has been performed
            time (Time): The time in which the simulaiton has been performed
            sensor (Sensor): The sensor mesuring in the simulaiton
            id (int): the ID of the sample

        Returns:
            str: path to the desired sample
        """
        return os.path.join(self._get_path_folder_CV(town, weather, time, sensor),
                            town.value+"_Opt_"+weather.value+time.value+"_"+str(id)+sensor.getEstention())

    def get_ids_CV(self, town: Town, weather: Weather, time: Time, sensor: Sensor):
        """get all the IDs of a specific simulation

        Args:
            town (Town): The town in which the simulaiton has been performed
            weather (Weather): The wether with which the simulaiton has been performed
            time (Time): The time in which the simulaiton has been performed
            sensor (Sensor): The sensor mesuring in the simulaiton

        Returns:
            list(int): list of all the available samples
        """
        folder = self._get_path_folder_CV(town, weather, time, sensor)
        folder_content = os.listdir(folder)
        folder_content = [file for file in folder_content if file.endswith(sensor.getEstention())]
        ids = [int(file.split('_')[3][:-4]) for file in folder_content]
        return ids
    
    def open_sample_CV(self, town: Town, weather: Weather, time: Time, sensor: Sensor, id:int):
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
            obj: the desired sample
        """
        file_path = self._get_path_file_CV(town, weather, time, sensor, id)
        if (sensor.getEstention() == ".ply"):
            data = PlyData.read(file_path)
            points = [data['vertex'][axis] for axis in ['x', 'y', 'z']]
            points = np.array(points).T
            objTag = data['vertex']['ObjTag']
            out = SelmaPointCloud(points, objTag)
        elif (sensor.getEstention() == ".jpg"):
            raise ValueError("Implementami")
        else:
            raise ValueError("Unkown file estention")
        return out

        
    