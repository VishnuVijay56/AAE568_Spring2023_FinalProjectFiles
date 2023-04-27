"""
autopilot_LQR_throttle_fault.py: autopilot designed to stabilize flight
                                 after known throttle fault
    - Author: Vishnu Vijay
    - Created: 4/26/23
"""

import numpy as np
from scipy.linalg import solve_continuous_are, solve_discrete_are

import control_parameters as AP
#from transfer_function import transferFunction
from wrap import wrap
import model_coef as M
from helper import QuaternionToEuler

from mav_state import MAV_State
from delta_state import Delta_State


class Autopilot_EF:
    def __init__(self, ts_control):
        # set time step
        self.Ts = ts_control

        # Trim state
        self.trim_d_e = M.u_trim.item(0)
        self.trim_d_a = M.u_trim.item(1)
        self.trim_d_r = M.u_trim.item(2)
        self.trim_d_t = M.u_trim.item(3)

        # Compute LQR gain
        # Lateral Autopilot
        A_lat = M.A_lat
        B_lat = M.B_lat
        
        q_v = 1e-1
        q_p = 1e0
        q_r = 1e-1
        q_phi = 1e3
        q_chi = 1e0
        Q_lat = np.diag([q_v, q_p, q_r, q_phi, q_chi])

        r_a = 1e1
        r_r = 1e1
        R_lat = np.diag([r_a, r_r])

        P_lat = solve_continuous_are(A_lat, B_lat, Q_lat, R_lat)
        self.K_lat = np.linalg.inv(R_lat) @ B_lat.T @ P_lat

        # Longitudinal Autopilot
        u_star = M.x_trim.item(3)
        w_star = M.x_trim.item(5)
        A_lon = M.A_lon
        B_lon = M.B_lon
        
        q_u = 1e2
        q_w = 1e2
        q_q = 0 #1e-2
        q_theta = 1e4
        q_h = 0
        Q_lon = np.diag([q_u, q_w, q_q, q_theta, q_h])

        r_e = 1e0
        r_t = 1e1
        R_lon = np.diag([r_e, r_t])

        P_lon = solve_continuous_are(A_lon, B_lon, Q_lon, R_lon)
        self.K_lon = np.linalg.inv(R_lon) @ B_lon.T @ P_lon

        self.commanded_state = MAV_State()


    def update(self, cmd, state):
        ## Lateral Autopilot

        err_Va = state.Va - cmd.airspeed_command

        chi_c = wrap(cmd.course_command, state.chi)
        err_chi = self.saturate(state.chi - chi_c, -np.radians(15), np.radians(15))

        x_lat = np.array([[err_Va * np.sin(state.beta)], #v
                          [state.p],
                          [state.r],
                          [state.phi],
                          [err_chi]], dtype=object)

        temp = -self.K_lat @ x_lat
        delta_a = self.saturate(temp.item(0) + self.trim_d_a, -np.radians(30), np.radians(30))
        delta_r = self.saturate(temp.item(1) + self.trim_d_r, -np.radians(30), np.radians(30))


        ## Longitudinal Autopilot

        alt_c = self.saturate(cmd.altitude_command, state.altitude - 0.2*AP.altitude_zone, state.altitude + 0.2*AP.altitude_zone)
        err_alt = state.altitude - alt_c
        err_down = -err_alt

        x_lon = np.array([[err_Va * np.cos(state.alpha)], # u
                          [err_Va * np.sin(state.alpha)], # w
                          [state.q], # q
                          [state.theta], # theta
                          [err_down]], dtype=object) # downward pos

        temp = -self.K_lon @ x_lon
        delta_e = self.saturate(temp.item(0) + self.trim_d_e, -np.radians(30), np.radians(30))
        delta_t = self.saturate((temp.item(1) + self.trim_d_t), 0., 1.)

        # construct output and commanded states
        delta = Delta_State(d_e = delta_e,
                            d_a = delta_a,
                            d_r = delta_r,
                            d_t = delta_t)
        self.commanded_state.altitude = cmd.altitude_command
        self.commanded_state.Va = cmd.airspeed_command
        self.commanded_state.phi = 0 # phi_c
        self.commanded_state.theta = 0 # theta_c
        self.commanded_state.chi = cmd.course_command

        return delta, self.commanded_state


    def saturate(self, input, low_limit, up_limit):
        if input <= low_limit:
            output = low_limit
        elif input >= up_limit:
            output = up_limit
        else:
            output = input
        return output
