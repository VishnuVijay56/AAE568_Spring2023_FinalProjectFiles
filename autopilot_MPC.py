"""
MPC Autopilot Block
"""

import numpy as np

from scipy.linalg import solve_continuous_are, solve_discrete_are
from casadi import *
import do_mpc

import control_parameters as AP
#from transfer_function import transferFunction
from wrap import wrap
import model_coef as M
from helper import QuaternionToEuler

from mav_state import MAV_State
from delta_state import Delta_State


class Autopilot:
    def __init__(self, ts_control, mpc_horizon):
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

        # Compute MPC gain
        '''
        1. Lateral State Definition
        2. Gain Definition
        3. MPC for lateral state definition
        '''
        # Initialize Lateral State Space
        H_lat = np.array([[0, 0, 0, 0, 1.0]])
        A_Alat = np.concatenate((np.concatenate((M.A_lat, np.zeros((5,1))), axis=1),
                                 np.concatenate((H_lat, np.zeros((1,1))), axis=1)), axis=0)
        B_Blat = np.concatenate((M.B_lat, np.zeros((1,2))), axis=0)

        # Q Gains
        q_v = 1e-1
        q_p = 1e0
        q_r = 1e-1
        q_phi = 1e0
        q_chi = 1e1
        q_int_chi = 0#1e-1
        Q_lat = np.diag([q_v, q_p, q_r, q_phi, q_chi, q_int_chi])

        # R Gains
        r_a = 1e1
        r_r = 1e0
        R_lat = np.diag([r_a, r_r])

        # Set up MPC Controller
        model_type = 'discrete'  # either 'discrete' or 'continuous'
        lateral_model = do_mpc.model.Model(model_type)

        # Initialize States
        _x_lat = lateral_model.set_variable(var_type='_x', var_name='x', shape=(np.shape(A_Alat, 1), 1))
        _u_lat = lateral_model.set_variable(var_type='_u', var_name='u', shape=(1, np.shape(B_Blat, 2)))

        # Define state update in MPC Toolbox
        x_next_lat = A_Alat@_x_lat + B_Blat@_u_lat
        lateral_model.set_rhs('x', x_next_lat)

        # Define cost function
        expression_lat = _x_lat.T@Q_lat@_x_lat + _u_lat.T@R_lat@ _u_lat

        # Build the model
        lateral_model.setup()

        # Define controller
        self.mpc_lat = do_mpc.controller.MPC(lateral_model)

        setup_lateral_mpc = {
            'n_robust': 0,
            'n_horizon': mpc_horizon,
            't_step': ts_control,
            'state_discretization': 'discrete',
            'store_full_solution':True,
            # Use MA27 linear solver in ipopt for faster calculations:
            #'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
        }

        self.mpc_lat.set_param(**setup_mpc)

        # Setting up terminal cost I think?
        mterm_lat = lateral_model.aux['cost']  # terminal cost
        lterm_lat = lateral_model.aux['cost']  # terminal cost
        self.mpc_lat.set_objective(mterm=mterm_lat, lterm=lterm_lat) # stage cost

        # This line is used in the toolbox
        #mpc.set_rterm(u=R_lat)  # input penalty

        # Constraints
        max_u_lat = np.array([[np.radians(30)], [np.radians(30)]])
        min_u_lat = -np.array([[np.radians(30)], [np.radians(30)]])

        max_u_lon = np.array([[np.radians(30)], [1.]])
        min_u_lon = np.array([[-np.radians(30)], [0.]])

        self.mpc_lat.bounds['lower', '_u', 'u'] = max_u_lat
        self.mpc_lat.bounds['lower', '_u', 'u'] = min_u_lat

        # Setup the mpc
        self.mpc_lat.setup()

        # We do not set initial guess, even though the toolbox does

        '''
        1. Longitudinal State
        2. Gain Definition
        3. MPC Setup
        '''
        # Longitudinal State Linearization
        u_star = M.x_trim.item(3)
        w_star = M.x_trim.item(5)
        H_lon = np.array([[0., 0., 0., 0., 1.], [u_star / AP.Va0, w_star / AP.Va0, 0., 0., 0.]])
        A_Alon = np.concatenate((np.concatenate((M.A_lon, np.zeros((5,2))), axis=1),
                                 np.concatenate((H_lon, np.zeros((2,2))), axis=1)), axis=0)
        B_Blon = np.concatenate((M.B_lon, np.zeros((2,2))), axis=0)

        # Longitudinal Q gains
        q_u = 1e1
        q_w = 1e1
        q_q = 1e-2
        q_theta = 1e-1
        q_h = 1e3
        q_int_h = 0#1e2
        q_int_Va = 0#1e2
        Q_lon = np.diag([q_u, q_w, q_q, q_theta, q_h, q_int_h, q_int_Va])

        # R gains
        r_e = 1e0
        r_t = 1e0
        R_lon = np.diag([r_e, r_t])




        '''State Definition'''
        self.commanded_state = MAV_State()


    def update(self, cmd, state):
        '''
        Lateral MPC
        '''

        err_Va = state.Va - cmd.airspeed_command

        chi_c = wrap(cmd.course_command, state.chi)
        err_chi = self.saturate(state.chi - chi_c, -np.radians(15), np.radians(15))

        self.int_course = self.int_course + (self.Ts / 2) * (err_chi + self.err_course_delay)
        self.err_course_delay = err_chi

        x_lat = np.array([[err_Va * np.sin(state.beta)],
                          [state.p],
                          [state.r],
                          [state.phi],
                          [err_chi],
                          [self.int_course]], dtype=object)

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
                          [self.int_Va]], dtype=object) # integral of airspeed

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