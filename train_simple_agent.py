from corr_study.agent.agent import *
from corr_study.simulation import *
from corr_study.datasetApi import *
import os

# for a, outfolder in zip ([2*10**-5, 4*10**-5, 6*10**-5], ["agent_out_a=2/", "agent_out_a=4/", "agent_out_a=6/"]):
a = 1*10**-5
outfolder = "agent_out_1e-5_full/"
print("alpha =", a)
ns = ["t3low", "t3medium", "t3high", "Town03LowNEW", "Town03MediumNEW", "Town03HighNEW"]
w = Weather.Clear
t = Time.Sunset
dataset = Dataset("corr_study/dataset/")
len_sim = 0
for n in ns:
    len_sim += dataset.get_measurement_series_length_TLC(n, w, t, Sensor.LT)

n_rep = 10
if not os.path.exists(outfolder):
    os.mkdir(outfolder)
agent = Agent(len_sim, n_rep, 
            3, 8, ["Top CD", "Left CD", "Right CD"], ["None", "Top", "Left", "Right", "TopLeft", "TopRight", "LeftRight", "All"],
            [[0, 1000], [0, 1000], [0, 1000]], 
            5*10**(-3), 10**-8, 0, 0.9, 32, 1000, 2000 )
eps = 0.5
for epoch in range(n_rep): 
    print("Epoch =", epoch)
    for n in ns:
        sim = Simulation(n, w, t, [Sensor.LT, Sensor.LFL, Sensor.LFR], 
                            mode=Simulation.RL_AGENT, train=True, agent=agent, 
                            alpha=a, out_folder=outfolder, visualize=False)
        sim.simulate(epsilon=eps)
        eps*=0.95
        print("Epsilon =", eps)
