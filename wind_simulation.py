"""
wind_simulation.py: Class to determine wind velocity at any given moment,
                    calculates a steady wind speed and uses a stochastic
                    process to represent wind gusts. (Follows section 4.4 in uav book)
    - Author: Vishnu Vijay
    - Created: 6/17/22
    - History:
        - 
"""

from transfer_function import transferFunction
import numpy as np

class WindSimulation:
    def __init__(self, Ts):
        # steady state wind defined in the inertial frame
        self._steady_state = np.array([[0., 0., 0.]]).T
        

        #   Dryden gust model parameters (section 4.4 UAV book)
        Va = 25 # must set Va to a constant value
        Lu = 200
        Lv = Lu
        Lw = 50
        gust_flag = False
        if gust_flag==False:
            sigma_u = 1.06
            sigma_v = sigma_u
            sigma_w = 0.7
        else:
            sigma_u = 2.12
            sigma_v = sigma_u
            sigma_w = 1.4

        # Dryden transfer functions (section 4.4 UAV book)
        self.u_w = transferFunction(num=np.array([[sigma_u*np.sqrt(2*Va)]]), 
                                    den=np.array([[np.sqrt(np.pi*Lu), Va*np.sqrt(np.pi/Lu)]]),
                                    Ts=Ts)
        self.v_w = transferFunction(num=np.array([[sigma_v*np.sqrt(3*Va), sigma_v*np.sqrt(Va**3)/Lv]]), 
                                    den=np.array([[np.sqrt(np.pi*Lv), np.sqrt(np.pi*Lv)*2*Va/Lv, np.sqrt(np.pi*Lv)*(Va**2)/(Lv**2)]]),
                                    Ts=Ts)
        self.w_w = transferFunction(num=np.array([[sigma_w*np.sqrt(3*Va), sigma_w*np.sqrt(Va**3)/Lw]]), 
                                    den=np.array([[np.sqrt(np.pi*Lw), np.sqrt(np.pi*Lw)*2*Va/Lw, np.sqrt(np.pi*Lw)*(Va**2)/(Lw**2)]]),
                                    Ts=Ts)
        self._Ts = Ts

    def update(self):
        # returns a six vector.
        #   The first three elements are the steady state wind in the inertial frame
        #   The second three elements are the gust in the body frame
        gust = np.array([[self.u_w.update(np.random.randn())],
                         [self.v_w.update(np.random.randn())],
                         [self.w_w.update(np.random.randn())]])

        return np.concatenate(( self._steady_state, gust ))