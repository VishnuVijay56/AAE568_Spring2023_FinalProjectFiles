"""
mav_state.py: class file for mav state
    - Author: Vishnu Vijay
    - Created: 6/2/22
    - History:
        - 6/7: Adding functionality for chapter 3
        - 6/16: Adding functionality for chapter 4
        - 7/14: Adding functionality for chapter 6

"""

import numpy as np
from wrap import wrap
import control_parameters as AP

class MAV_State:
    def __init__(self, *args):
        if (len(args) == 0):
            # Inertial Position
            self.north = 0
            self.east = 0
            self.altitude = 0
            
            # Angular Positions
            self.phi = 0 # roll in radians
            self.theta = 0 # pitch in radians
            self.psi = 0 # heading in radians

            # Rate of Change of Angular Positions
            self.p = 0 # roll rate in rad/s
            self.q = 0 # pitch rate in rad/s
            self.r = 0 # heading rate in rad/s

            # Flight Parameters
            self.Va = 0 # airspeed
            self.alpha = 0 # angle of attack
            self.beta = 0 # sideslip angle
            self.Vg = 0 # groundspeed
            self.gamma = 0 # flight path angle
            self.chi = 0 # course angle

            # Wind
            self.wn = 0 # inertial wind north
            self.we = 0 # inertial wind east
        else:
            # Inertial Position
            self.north = 0
            self.east = 0
            self.altitude = args[0]
            
            # Angular Positions
            self.phi = args[1] # roll in radians
            self.theta = args[2] # pitch in radians
            self.psi = args[3] # heading in radians

            # Rate of Change of Angular Positions
            self.p = args[4] # roll rate in rad/s
            self.q = args[5] # pitch rate in rad/s
            self.r = args[6] # heading rate in rad/s

            # Flight Parameters
            self.Va = args[7] # airspeed
            self.alpha = 0 # angle of attack
            self.beta = 0 # sideslip angle
            self.Vg = 0 # groundspeed
            self.gamma = 0 # flight path angle
            self.chi = 0 # course angle

            # Wind
            self.wn = 0 # inertial wind north
            self.we = 0 # inertial wind east


    def get_lat_state(self):
        #err_Va = self.Va - cmd.airspeed_command

        #chi_c = wrap(cmd.course_command, self.chi)
        #err_chi = self.saturate(self.chi - chi_c, -np.radians(15), np.radians(15))

        x_lat = np.array([[float(self.Va * np.sin(self.beta))],
                          [float(self.p)],
                          [float(self.r)],
                          [float(self.phi)],
                          [float(self.chi)]])
        return x_lat

    
    def get_lon_state(self):
        #err_Va = self.Va - cmd.airspeed_command

        #alt_c = self.saturate(cmd.altitude_command, self.altitude - 0.2*AP.altitude_zone, self.altitude + 0.2*AP.altitude_zone)
        #err_alt = self.altitude - alt_c
        #err_down = -err_alt
        

        x_lon = np.array([[float(self.Va * np.cos(self.alpha))], # u
                          [float(self.Va * np.sin(self.alpha))], # w
                          [float(self.q)], # q
                          [float(self.theta)], # theta
                          [-float(self.altitude)]]) # alt
        return x_lon


    def set_initial_cond(self, x_lon, x_lat):
        # Velocity
        self.Va = np.sqrt(x_lon.item(0)**2 + x_lon.item(1)**2 + x_lat.item(0)**2)
        
        # angular rate
        self.p = x_lat.item(1)
        self.q = x_lon.item(2)
        self.r = x_lat.item(2)

        # orientation
        self.phi = x_lat.item(3)
        self.theta = x_lon.item(3)
        self.psi = x_lat.item(4)

        # alt
        self.altitude = x_lon.item(4)

    def print(self):
        print("MAV STATE:")
        print("\tNorth: {}; East: {}; Alt: {}".format(self.north, self.east, self.altitude))
        print("\tPhi: {}; Theta: {}; Psi: {}".format(self.phi, self.theta, self.psi))
        print("\tP: {}; Q: {}; R: {}".format(self.p, self.q, self.r))
        print("\tAoA: {}; Beta: {}; Gamma: {}; Chi: {}; Va: {}".format(self.alpha, self.beta, self.gamma, self.chi, self.Va))

    
    def saturate(self, input, low_limit, up_limit):
        if input <= low_limit:
            output = low_limit
        elif input >= up_limit:
            output = up_limit
        else:
            output = input
        return output