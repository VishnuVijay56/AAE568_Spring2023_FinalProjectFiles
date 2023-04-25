"""
AAE568_project.py: implementing project
    - Author: Vishnu Vijay
    - Created: 4/23/23
"""

# Imports
from run_sim import run_two_plane_sim
from sim_cmds import SimCmds 


# Simulation Parameters
sim_opt = SimCmds()
sim_opt.view_sim = False
sim_opt.sim_real_time = False
sim_opt.display_graphs = True
sim_opt.use_kf = False
sim_opt.wind_gust = False

# Time Span
t_span = (0, 20)

# Sim
run_two_plane_sim(t_span, sim_opt)