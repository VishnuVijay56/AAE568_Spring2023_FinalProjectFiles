"""
AAE568_project.py: implementing project
    - Author: Vishnu Vijay
    - Created: 4/23/23
"""

# Imports
from run_sim import run_two_plane_sim
from sim_cmds import SimCmds 
from saturate_cmds import SaturateCmds


# Simulation Parameters
sim_opt = SimCmds()
sim_opt.view_sim = False
sim_opt.sim_real_time = False
sim_opt.display_graphs = False
sim_opt.use_kf = True
sim_opt.wind_gust = False

# from helper import write_discrete_SS
# write_discrete_SS()

# Time Span
t_span = (0, 40)


# Sim
for i in range(15):
    print("----------------------------------")
    print("Simulation number: ", i)
    run_two_plane_sim(t_span, sim_opt)