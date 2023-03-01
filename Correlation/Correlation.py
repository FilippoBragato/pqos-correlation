import numpy as np

def comupte_correlation(x:dict,y:dict):
    if x["type"] == "Voxels" and y["type"] == "Voxels":
        cross_energy = (np.sum(x["data"]**2) * np.sum(y["data"]**2))**0.5
        return np.sum(x["data"]*y["data"])/cross_energy
    else:
        raise ValueError("Unsupported type or type missmatch between data") 