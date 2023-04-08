"""
LQR autopilot block for mavsim_python
    - Beard & McLain, PUP, 2012
    - Last Update:
        2/6/2019 - RWB
"""

import numpy as np
from scipy.linalg import solve_continuous_are, solve_discrete_are

import control_parameters as AP
#from transfer_function import transferFunction
from wrap import wrap
from pi_control import PIControl
from pd_control_with_rate import PDControlWithRate
import model_coef as M
from helper import QuaternionToEuler

from mav_state import MAV_State
from delta_state import Delta_State


class Autopilot:
    def __init__(self, ts_control):
        # set time step
        self.Ts = ts_control

        # Trim state
        self.trim_d_e = M.u_trim.item(0)
        self.trim_d_a = M.u_trim.item(1)
        self.trim_d_r = M.u_trim.item(2)
        self.trim_d_t = M.u_trim.item(3)

        # Initialize integrators and delay vars
        self.int_course = 0
        self.int_down = 0
        self.int_Va = 0
        self.err_course_delay = 0
        self.err_down_delay = 0
        self.err_Va_delay = 0

        # Compute LQR gain
        # Lateral Autopilot
        H_lat = np.array([[0, 0, 0, 0, 1.0]])
        A_Alat = np.concatenate((np.concatenate((M.A_lat, np.zeros((5,1))), axis=1),
                                 np.concatenate((H_lat, np.zeros((1,1))), axis=1)), axis=0)
        B_Blat = np.concatenate((M.B_lat, np.zeros((1,2))), axis=0)
        
        q_v = 1e-1
        q_p = 1e0
        q_r = 1e-1
        q_phi = 1e0
        q_chi = 1e1
        q_int_chi = 0#1e-1
        Q_lat = np.diag([q_v, q_p, q_r, q_phi, q_chi, q_int_chi])
        # Q_lat = np.diag([0.001, 0.01, 0.1, 100, 1, 100])

        r_a = 1e1
        r_r = 1e0
        R_lat = np.diag([r_a, r_r])

        P_lat = solve_continuous_are(A_Alat, B_Blat, Q_lat, R_lat)
        self.K_lat = np.linalg.inv(R_lat) @ B_Blat.T @ P_lat

        # Longitudinal Autopilot
        u_star = M.x_trim.item(3)
        w_star = M.x_trim.item(5)
        H_lon = np.array([[0., 0., 0., 0., 1.], [u_star / AP.Va0, w_star / AP.Va0, 0., 0., 0.]])
        A_Alon = np.concatenate((np.concatenate((M.A_lon, np.zeros((5,2))), axis=1),
                                 np.concatenate((H_lon, np.zeros((2,2))), axis=1)), axis=0)
        B_Blon = np.concatenate((M.B_lon, np.zeros((2,2))), axis=0)
        
        q_u = 1e1
        q_w = 1e1
        q_q = 1e-2
        q_theta = 1e-1
        q_h = 1e3
        q_int_h = 0#1e2
        q_int_Va = 0#1e2
        Q_lon = np.diag([q_u, q_w, q_q, q_theta, q_h, q_int_h, q_int_Va])
        # Q_lon = np.diag([10, 10, 0.001, 0.01, 10, 100, 100])

        r_e = 1e0
        r_t = 1e0
        R_lon = np.diag([r_e, r_t])

        P_lon = solve_continuous_are(A_Alon, B_Blon, Q_lon, R_lon)
        self.K_lon = np.linalg.inv(R_lon) @ B_Blon.T @ P_lon

        # eigvals_lon, eigvects_lon = np.linalg.eig(M.A_lon - M.B_lon @ self.K_lon)
        # eigvals_lat, eigvects_lat = np.linalg.eig(M.A_lat - M.B_lat @ self.K_lat)

        # print("LON:\n", eigvals_lon)
        # print("\nLAT:\n", eigvals_lat)

        print("K_lon: \n", self.K_lon)
        print("\nK_lat: \n", self.K_lat)

        self.commanded_state = MAV_State()


    def update(self, cmd, state):
        # Quaternion orientation to euler
        #trim_phi, trim_theta, trim_psi = QuaternionToEuler(M.x_trim[6:10])

        # Lateral Autopilot
        err_Va = state.Va - cmd.airspeed_command

        chi_c = wrap(cmd.course_command, state.chi)
        err_chi = self.saturate(state.chi - chi_c, -np.radians(15), np.radians(15))

        self.int_course = self.int_course + (self.Ts / 2) * (err_chi + self.err_course_delay)
        self.err_course_delay = err_chi

        #self.int_course = self.int_course + 

        x_lat = np.array([[err_Va * np.sin(state.beta)],
                          [state.p],
                          [state.r],
                          [state.phi],
                          [err_chi],
                          [self.int_course]])

        temp = -self.K_lat @ x_lat
        delta_a = self.saturate(temp.item(0) + self.trim_d_a, -np.radians(30), np.radians(30))
        delta_r = self.saturate(temp.item(1) + self.trim_d_r, -np.radians(30), np.radians(30))


        # Longitudinal Autopilot
        alt_c = self.saturate(cmd.altitude_command, state.altitude - 0.2*AP.altitude_zone, state.altitude + 0.2*AP.altitude_zone)
        err_alt = state.altitude - alt_c
        err_down = -err_alt
        
        self.int_down = self.int_down + (self.Ts / 2) * (err_down + self.err_down_delay)
        self.err_down_delay = err_down
        
        self.int_Va = self.int_Va + (self.Ts / 2) * (err_Va + self.err_Va_delay)
        self.err_Va_delay = err_Va

        x_lon = np.array([[err_Va * np.cos(state.alpha)], # u
                          [err_Va * np.sin(state.alpha)], # w
                          [state.q], # q
                          [state.theta], # theta
                          [err_down], # downward pos
                          [self.int_down], # integral of altitude
                          [self.int_Va]]) # integral of airspeed

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
