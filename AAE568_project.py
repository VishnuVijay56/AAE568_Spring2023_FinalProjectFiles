"""
AAE568_project.py: implementing project
    - Author: Vishnu Vijay
    - Created: 4/23/23
"""
import numpy as np

# Imports
from run_sim import run_two_plane_sim
from sim_cmds import SimCmds 


# Simulation Parameters
sim_opt = SimCmds()
sim_opt.view_sim = False
sim_opt.sim_real_time = False
sim_opt.display_graphs = False
sim_opt.use_kf = False
sim_opt.wind_gust = False

# Time Span
t_span = (0, 40)

# Sim
num = 15
data = np.zeros([num, 5, 1])
for i in range(num):
    point_data = run_two_plane_sim(t_span, sim_opt)
    data[i, :] = point_data
    # print(i+1)
print("##########################")

results = np.mean(data, 0)
results[0, 0] = np.rad2deg(results[0, 0])
print(results)
