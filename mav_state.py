"""
mav_state.py: class file for mav state
    - Author: Vishnu Vijay
    - Created: 6/2/22
    - History:
        - 6/7: Adding functionality for chapter 3
        - 6/16: Adding functionality for chapter 4
        - 7/14: Adding functionality for chapter 6

"""

class MAV_State:
    def __init__(self):
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


    
    def print(self):
        print("MAV STATE:")
        print("\tNorth: {}; East: {}; Alt: {}".format(self.north, self.east, self.altitude))
        print("\tPhi: {}; Theta: {}; Psi: {}".format(self.phi, self.theta, self.psi))
        print("\tP: {}; Q: {}; R: {}".format(self.p, self.q, self.r))
        print("\tAoA: {}; Beta: {}; Gamma: {}; Chi: {}; Va: {}".format(self.alpha, self.beta, self.gamma, self.chi, self.Va))