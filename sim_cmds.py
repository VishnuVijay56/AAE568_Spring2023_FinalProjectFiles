"""
sim_cmds.py: command/message type for simulation
    - Author: Vishnu Vijay
    - Created: 4/25/23
"""

class SimCmds:
    def __init__(self):
        self.view_sim = False
        self.sim_real_time = False
        self.display_graphs = False
        self.use_kf = False
        self.wind_gust = False