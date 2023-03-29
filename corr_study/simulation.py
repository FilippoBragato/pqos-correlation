from . import datasetApi
import h5py
import numpy as np
from . import datasetApi
import open3d as o3d
import seaborn as sns
import copy
from tqdm import trange

class Simulation():

    ## DECIDE IF TRANSMIT
    MAX_CHAMFER_DISTANCE = 1

    def __init__(self, simulation_name, weather, time, sensors, dataset_location='corr_study/dataset/', mode=MAX_CHAMFER_DISTANCE, verbose=False, visualize=False) -> None:
        self.dataset = datasetApi.Dataset(dataset_location)
        self.frame_idx = 1
        self.name = simulation_name
        self.weather = weather
        self.time = time
        self.sensors = sensors
        self.visualize = visualize
        self.length = self.dataset.get_measurement_series_length_TLC(self.name, self.weather, self.time, self.sensors[0])
        self.transmissions = np.zeros((len(self), len(self.sensors)), dtype=bool)
        self.mode = mode
        self.verbose = verbose
        self.ego_positions = self._open_positions()
        self.opened_samples = {}
        self.ignore_ego = True
        for s in self.sensors:
            self.opened_samples[s] = []
        self.n_sample_to_open = 100
        if self.verbose:
            print("Initialized simulation", self.name)
            print("Weather =", self.weather)
            print("Time =", self.time)
            print("Sensors =", self.sensors, "\n")

    def _open_positions(self):
        path_to_bbox = self.dataset.get_path_bbox(self.name, self.weather, self.time)
        with h5py.File(path_to_bbox,'r') as f:
            root_grp = f.get("BBOX")
            ids = list(root_grp.keys())
            ids.sort()
            for id_actor in ids:
                agent = root_grp.get(id_actor)
                if "vehicle" in agent.attrs['type']:
                    ego = root_grp.get(id_actor)
                    self.ego_id = int(id_actor)
                    break
            loc = np.array(ego.get('location'))
            rot = np.array(ego.get('rotation'))
            if self.verbose:
                print("Ego vehicle found as", ego, "\n")
        return {"loc":loc, "rot":rot}
    
    def go_next_frame(self):
        if self.visualize:
            self.frame_idx = self.frame_idx + 1
        else:
            self.frame_idx = self.frame_idx + 1

        if self.verbose == 2:
            print("Going to frame", self.frame_idx, "\n")

    def _open_or_read_sample(self, route:str, weather:datasetApi.Weather, time:datasetApi.Time, sensor:datasetApi.Sensor, time_step:int):
        if len(self.opened_samples[sensor]) == 0:
            # Open n samples
            samples = self.dataset.open_measurement_samples_TLC(route, weather, time, sensor, list(range(time_step, time_step + self.n_sample_to_open)))
            self.opened_samples[sensor] += samples
        return self.opened_samples[sensor].pop(0)

    
    def open_this_frame_measurements(self):
        data = []
        self.frame_position = [-self.ego_positions["loc"][self.frame_idx-1,1], 
                               -self.ego_positions["loc"][self.frame_idx-1,0], 
                                self.ego_positions["loc"][self.frame_idx-1,2]]
        for sensor in self.sensors:
            pcs = self.dataset.open_measurement_sample_TLC(self.name, self.weather, self.time, sensor, self.frame_idx)
            #pcs = self._open_or_read_sample(self.name, self.weather, self.time, sensor, self.frame_idx)
            if self.ignore_ego:
                pcs.data = pcs.data[pcs.ground_truth[:,1] != self.ego_id,:]
                pcs.ground_truth = pcs.ground_truth[pcs.ground_truth[:,1] != self.ego_id,:]

            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(pcs.data)
                

            T_sens = sensor.get_homogeneous_matrix()
            T_rot = np.eye(4)
            T_rot[:3, :3] = pc.get_rotation_matrix_from_xyz((np.radians(-self.ego_positions["rot"][self.frame_idx-1,2]),
                                                             np.radians(-self.ego_positions["rot"][self.frame_idx-1,0]), 
                                                             np.radians(-self.ego_positions["rot"][self.frame_idx-1,1])))
            T_loc = np.eye(4)
            T_loc[0:3,3] = [-self.ego_positions["loc"][self.frame_idx-1,1],
                            -self.ego_positions["loc"][self.frame_idx-1,0],
                             self.ego_positions["loc"][self.frame_idx-1,2]]
            T = np.dot(T_loc, T_rot)
            T = np.dot(T, T_sens)
            pc.transform(T)

            data.append(pc)
            if self.verbose == 2:
                print("Opening sample", self.frame_idx, "for", sensor)
        if self.verbose == 2:
            print("")
        return np.array(data)
    
    def __len__(self):
        if self.visualize:
            return 1000#int(np.floor(self.length/100))
        return 100#self.length
    
    def decide_if_transmit(self):
        PAVEMENT_HEIGHT = 0.6
        if self.mode == Simulation.MAX_CHAMFER_DISTANCE:
            MAX_DISTANCE_CONSIDERED = 30
            if not hasattr(self, "last_sent"):
                self.list_last_sent = np.array([None]*len(self.sensors))
                return [True] * len(self.sensors)
            else:
                out = []
                for sample in self.frame_data:
                    last_sent_array = np.asarray(self.last_sent.points)

                    last_sent_mask = (((last_sent_array[:,0] - self.frame_position[0]) **2 + 
                                       (last_sent_array[:,1]- self.frame_position[1]) **2) < MAX_DISTANCE_CONSIDERED**2)
                    last_sent_mask = np.logical_and(last_sent_mask, last_sent_array[:,2] > self.frame_position[2] + PAVEMENT_HEIGHT)
                    last_sent_cropped = self.last_sent.select_by_index(np.where(last_sent_mask)[0])
                    if last_sent_mask.sum == 0:
                        print("Mandato vuoto")

                    sample_cropped_array = np.asarray(sample.points)
                    sample_cropped_mask = (((sample_cropped_array[:,0] - self.frame_position[0]) **2 + 
                                            (sample_cropped_array[:,1]- self.frame_position[1]) **2) < MAX_DISTANCE_CONSIDERED**2)
                    sample_cropped_mask = np.logical_and(sample_cropped_mask, sample_cropped_array[:,2] > self.frame_position[2] + PAVEMENT_HEIGHT)
                    sample_cropped = sample.select_by_index(np.where(sample_cropped_mask)[0])
                    if sample_cropped_mask.sum == 0:
                        print("Campione vuoto")
                    # d1 = last_sent_cropped.compute_point_cloud_distance(sample_cropped)
                    d2 = sample_cropped.compute_point_cloud_distance(last_sent_cropped)
                    # cd = np.sum(np.asarray(d1)**2) + np.sum(np.asarray(d2)**2)
                    sum_of_distance = np.sum(np.asarray(d2))
                    if self.verbose:
                        print("Found distance", sum_of_distance)
                    # sample_cropped.paint_uniform_color([0,1,0])
                    # last_sent_cropped.paint_uniform_color([0,0,1])
                    # o3d.visualization.draw_geometries([sample_cropped, last_sent_cropped])
                    out.append(sum_of_distance > 500)
        else:
            raise NotImplementedError("Ehm... ")
        if self.verbose:
            print("It has been decided to transmit", out, "\n")
        return out
    
    def transmit(self, transmit):
        actual_transmitted_stuff = np.zeros_like(self.list_last_sent)
        actual_transmitted_stuff[transmit] = self.frame_data[transmit]
        actual_transmitted_stuff[np.logical_not(transmit)] = self.list_last_sent[np.logical_not(transmit)]
        self.transmissions[self.frame_idx-1, :] = transmit
        pcd = o3d.geometry.PointCloud()
        for td in actual_transmitted_stuff:
            pcd = pcd + td
        self.last_sent = pcd
        self.list_last_sent = actual_transmitted_stuff 

    def simulate(self):
        if self.visualize:
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis_pcd = o3d.geometry.PointCloud(self.open_this_frame_measurements()[0].points)
            vis.add_geometry(vis_pcd)
        for _ in trange(self.frame_idx, len(self)):
            self.frame_data = self.open_this_frame_measurements()
            transmit = self.decide_if_transmit()
            self.transmit(transmit)
            if self.visualize:
                
                vis_pcd.points = self.last_sent.points
                vis.update_geometry(vis_pcd)
                vis.poll_events()
                vis.update_renderer()
            self.go_next_frame()

