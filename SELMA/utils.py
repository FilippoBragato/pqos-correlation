import open3d as o3d
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
def visualize_sample(sample:dict):
    """visualize the sample

    Args:
        sample (dict): the sample to visualize 

    Raises:
        ValueError: if the function does not know how to handle the format
    """
    
    if (sample["type"] == "PointCloud"):

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(sample["data"])

        if "class" in sample.keys():
            palette = sns.color_palette("hsv", n_colors=36)
            get_color = lambda tag:palette[tag%36]
            colors = np.array(np.vectorize(get_color)(sample["class"])).T

            pcd.colors = o3d.utility.Vector3dVector(colors)

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.get_render_option().background_color = np.asarray([0, 0, 0])
        vis.run()
        vis.destroy_window()

    elif (sample["type"] == "Image"):
        
        plt.imshow(sample["data"][:,:,[2,1,0]])
        plt.show()

    else:
        raise ValueError("Unkown file type")