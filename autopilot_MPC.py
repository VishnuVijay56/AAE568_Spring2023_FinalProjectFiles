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
    def __init__(self, ts_control, mpc_horizon, state):
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

        suppress_ipopt = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}

        # Compute MPC gain
        '''
        1. Lateral State Definition
        2. Gain Definition
        3. MPC for lateral state definition
        '''
        # Initialize Lateral State Space
        A_Alat = M.A_lat
        B_Blat = M.B_lat

        # H_lat = np.array([[0, 0, 0, 0, 1.0]])
        # A_Alat = np.concatenate((np.concatenate((M.A_lat, np.zeros((5,1))), axis=1),
        #                          np.concatenate((H_lat, np.zeros((1,1))), axis=1)), axis=0)
        # B_Blat = np.concatenate((M.B_lat, np.zeros((1,2))), axis=0)

        # Q Gains
        q_v = 1e-1
        q_p = 1e0
        q_r = 1e-1
        q_phi = 1e0
        q_chi = 1e1
        Q_lat = np.diag([q_v, q_p, q_r, q_phi, q_chi])

        # R Gains
        r_a = 1e1
        r_r = 1e0
        R_lat = np.array([[r_a], [r_r]])  # Do-MPC does not like R as a matrix. Instead, it wants one "input penalty"
        # for each input

        ###
        # Start Defining Lateral MPC
        ###

        # Set up MPC Controller
        model_type = 'discrete'  # either 'discrete' or 'continuous'
        lateral_model = do_mpc.model.Model(model_type)

        # Initialize States
        _x_lat = lateral_model.set_variable(var_type='_x', var_name='x', shape=(np.shape(A_Alat)[0], 1))
        _u_lat = lateral_model.set_variable(var_type='_u', var_name='u', shape=(np.shape(B_Blat)[1], 1))

        # Define state update in MPC Toolbox
        x_next_lat = A_Alat@_x_lat + B_Blat@_u_lat
        lateral_model.set_rhs('x', x_next_lat)

        # Define cost function
        expression_lat = _x_lat.T@Q_lat@_x_lat  #  + _u_lat.T@R_lat@ _u_lat
        lateral_model.set_expression(expr_name='cost', expr=expression_lat)

        # Build the model
        lateral_model.setup()

        ## Define controller
        self.mpc_lat = do_mpc.controller.MPC(lateral_model)

        # General Settings
        setup_lateral_mpc = {
            'n_robust': 0,
            'n_horizon': mpc_horizon,
            't_step': ts_control,
            'state_discretization': 'discrete',
            'store_full_solution':True,
            # Use MA27 linear solver in ipopt for faster calculations:
            #'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
        }

        self.mpc_lat.set_param(**setup_lateral_mpc, nlpsol_opts=suppress_ipopt)

        # Setting up terminal cost I think?
        mterm_lat = lateral_model.aux['cost']  # terminal cost
        lterm_lat = lateral_model.aux['cost']  # terminal cost
        self.mpc_lat.set_objective(mterm=mterm_lat, lterm=lterm_lat) # stage cost

        # This line is used in the toolbox
        self.mpc_lat.set_rterm(u=R_lat)  # input penalty

        # Constraints
        max_u_lat = np.array([[np.radians(30)], [np.radians(30)]])
        min_u_lat = -np.array([[np.radians(30)], [np.radians(30)]])

        self.mpc_lat.bounds['upper', '_u', 'u'] = max_u_lat
        self.mpc_lat.bounds['lower', '_u', 'u'] = min_u_lat

        # Scaling
        scaling_array_lat = np.array([1/28, 1, 1, 1, 1])
        self.mpc_lat.scaling['_x', 'x'] = scaling_array_lat

        # Setup the mpc
        self.mpc_lat.setup()

        # Initialize initial conditions
        self.mpc_lat.x0 = state.get_lat_state()
        self.mpc_lat.set_initial_guess()

        '''
        1. Longitudinal State
        2. Gain Definition
        3. MPC Setup
        '''
        # Longitudinal State Linearization
        A_Alon = M.A_lon
        B_Blon = M.B_lon

        # u_star = M.x_trim.item(3)
        # w_star = M.x_trim.item(5)
        # H_lon = np.array([[0., 0., 0., 0., 1.], [u_star / AP.Va0, w_star / AP.Va0, 0., 0., 0.]])
        # A_Alon = np.concatenate((np.concatenate((M.A_lon, np.zeros((5,2))), axis=1),
        #                          np.concatenate((H_lon, np.zeros((2,2))), axis=1)), axis=0)
        # B_Blon = np.concatenate((M.B_lon, np.zeros((2,2))), axis=0)

        # Longitudinal Q gains
        q_u = 1e1
        q_w = 1e1
        q_q = 1e-2
        q_theta = 1e-1
        q_h = 1e3
        Q_lon = np.diag([q_u, q_w, q_q, q_theta, q_h])

        # R gains
        r_e = 1e0
        r_t = 1e0
        R_lon = np.array([[r_e], [r_t]])

        ###
        # Start Defining Longitudinal MPC
        ###

        # Set up MPC Controller
        model_type = 'discrete'  # either 'discrete' or 'continuous'
        longitudinal_model = do_mpc.model.Model(model_type)

        # Initialize States
        _x_lon = longitudinal_model.set_variable(var_type='_x', var_name='x', shape=(np.shape(A_Alon)[0], 1))
        _u_lon = longitudinal_model.set_variable(var_type='_u', var_name='u', shape=(np.shape(B_Blon)[1], 1))

        # Define state update in MPC Toolbox
        x_next_lon = A_Alon @ _x_lon + B_Blon @ _u_lon
        longitudinal_model.set_rhs('x', x_next_lon)

        # Define cost function
        expression_lon = _x_lon.T @ Q_lon @ _x_lon  # + _u_lon.T @ R_lon @ _u_lon
        longitudinal_model.set_expression(expr_name='cost', expr=expression_lon)
        # NOTS: Toolbox defines R differently

        # Build the model
        longitudinal_model.setup()

        # Define controller
        self.mpc_lon = do_mpc.controller.MPC(longitudinal_model)

        setup_longitudinal_mpc = {
            'n_robust': 0,
            'n_horizon': mpc_horizon,
            't_step': ts_control,
            'state_discretization': 'discrete',
            'store_full_solution': True,
            # Use MA27 linear solver in ipopt for faster calculations:
            # 'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
        }

        self.mpc_lon.set_param(**setup_longitudinal_mpc, nlpsol_opts=suppress_ipopt)

        # Setting up terminal cost I think?
        mterm_lon = longitudinal_model.aux['cost']  # terminal cost
        lterm_lon = longitudinal_model.aux['cost']  # terminal cost
        self.mpc_lon.set_objective(mterm=mterm_lon, lterm=lterm_lon)  # stage cost

        # This line is used in the toolbox
        self.mpc_lon.set_rterm(u=R_lon)  # input penalty

        # Constraints
        max_u_lon = np.array([[np.radians(30)], [1.]])
        min_u_lon = np.array([[-np.radians(30)], [0.]])

        self.mpc_lon.bounds['upper', '_u', 'u'] = max_u_lon
        self.mpc_lon.bounds['lower', '_u', 'u'] = min_u_lon

        # Scaling
        scaling_array_lon = np.array([1, 1, 1, 1, 1/15])
        self.mpc_lon.scaling['_x', 'x'] = scaling_array_lon

        # Setup the mpc
        self.mpc_lon.setup()

        # Initialize initial conditions
        self.mpc_lon.x0 = state.get_lon_state()
        self.mpc_lon.set_initial_guess()
        # self.mpc_lon.supress_ipopt_output(self)
        # self.mpc_lat.supress_ipopt_output(self)

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
                          [err_chi]], dtype=object)
        lat_control = self.mpc_lat.make_step(x_lat)
        delta_a = lat_control[0, 0]
        delta_r = lat_control[1, 0]

        # delta_a = self.saturate(control.item(0) + self.trim_d_a, -np.radians(30), np.radians(30))
        # delta_r = self.saturate(control.item(1) + self.trim_d_r, -np.radians(30), np.radians(30))


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
                          [err_down]], dtype=object)  # downward position

        lon_control = self.mpc_lat.make_step(x_lat)
        delta_e = lon_control[0, 0]
        delta_t = lat_control[1, 0]

        # delta_e = self.saturate(control.item(0) + self.trim_d_e, -np.radians(30), np.radians(30))
        # delta_t = self.saturate((control.item(1) + self.trim_d_t), 0., 1.)

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