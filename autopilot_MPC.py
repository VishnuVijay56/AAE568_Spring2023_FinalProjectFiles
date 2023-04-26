"""
MPC Autopilot Block
"""

import numpy as np

from scipy.linalg import solve_continuous_are, solve_discrete_are
from casadi import *
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc

import control_parameters as AP
from wrap import wrap
import model_coef as M

from mav_state import MAV_State
from delta_state import Delta_State


class Autopilot_MPC:
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

        # Supress ipopt Output (used in General Settings)
        suppress_ipopt = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}

        # Matrix definition
        A_Alon = np.array([
            [0.997908, 0.005126, -0.011762, -0.097877, 0.000000],
            [-0.005237, 0.951797, 0.231706, -0.005495, 0.000000],
            [0.002051, -0.037957, 0.943900, 0.000011, 0.000000],
            [0.000010, -0.000190, 0.009717, 1.000000, 0.000000],
            [-0.000528, 0.009769, -0.000055, -0.250096, 1.000000]])
        B_Blon = np.array([
            [0.000677, 0.081990],
            [-0.067072, -0.000215],
            [-0.350505, 0.000084],
            [-0.001752, 0.000000],
            [-0.000116, -0.000022]])
        A_Alat = np.array([
            [0.991075, 0.011764, -0.246394, 0.097539, 0.000000],
            [-0.034201, 0.796460, 0.101598, -0.001675, 0.000000],
            [0.007768, -0.000982, 0.986781, 0.000381, 0.000000],
            [-0.000169, 0.008982, 0.001006, 0.999992, 0.000000],
            [0.000039, -0.000005, 0.009946, 0.000002, 1.000000]])
        B_Blat = np.array([
            [0.016320, 0.068029],
            [1.177928, -0.029419],
            [0.049201, -0.247014],
            [0.005902, -0.000209],
            [0.000246, -0.001237]])
        '''
        1. Lateral State Definition
        2. Gain Definition
        3. MPC for lateral state definition
        '''
        # Initialize Lateral State Space
        # A_Alat = M.A_lat
        # B_Blat = M.B_lat

        # Q Gains
        q_v = 1e1
        q_p = 1e0
        q_r = 1e-1
        q_phi = 1e0
        q_chi = 1e1
        Q_lat = np.diag([q_v, q_p, q_r, q_phi, q_chi])

        # R Gains
        r_a = 1e1
        r_r = 1e0
        R_lat = np.array([[r_a], [r_r]])  # Do-MPC does not like R as a matrix. Instead, it wants one "input penalty"
        # for each input, so for this case it is a column vector

        ###
        # Start Defining Lateral MPC
        ###

        # Set up MPC Controller
        model_type = 'discrete'  # either 'discrete' or 'continuous'
        lateral_model = do_mpc.model.Model(model_type)

        # Initialize States
        _x_lat = lateral_model.set_variable(var_type='_x', var_name='x_lat', shape=(np.shape(A_Alat)[0], 1))
        _u_lat = lateral_model.set_variable(var_type='_u', var_name='u_lat', shape=(np.shape(B_Blat)[1], 1))

        # Define state update in MPC Toolbox
        x_next_lat = A_Alat@_x_lat + B_Blat@_u_lat
        lateral_model.set_rhs('x_lat', x_next_lat)

        # Define cost function
        expression_lat = _x_lat.T@Q_lat@_x_lat
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
        }

        self.mpc_lat.set_param(**setup_lateral_mpc, nlpsol_opts=suppress_ipopt)

        # Setting up terminal cost I think?
        # mterm_lat = lateral_model.aux['cost']  # terminal cost
        mterm_lat = DM(1, 1)
        lterm_lat = lateral_model.aux['cost']  # terminal cost
        self.mpc_lat.set_objective(mterm=mterm_lat, lterm=lterm_lat)  # stage cost

        # Set Control Cost
        self.mpc_lat.set_rterm(u_lat=R_lat)  # input penalty

        # Constraints
        max_u_lat = np.array([[np.radians(30)], [np.radians(30)]])
        min_u_lat = -np.array([[np.radians(30)], [np.radians(0)]])

        self.mpc_lat.bounds['upper', '_u', 'u_lat'] = max_u_lat
        self.mpc_lat.bounds['lower', '_u', 'u_lat'] = min_u_lat

        # Scaling
        scaling_array_lat = np.array([1, 1, 1, 1, 1])
        self.mpc_lat.scaling['_x', 'x_lat'] = scaling_array_lat

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
        # A_Alon = M.A_lon
        # B_Blon = M.B_lon

        # Longitudinal Q gains
        q_u = 1e2
        q_w = 1e2
        q_q = 1e-2
        q_theta = 1e-1
        q_h = 1e4
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
        _x_lon = longitudinal_model.set_variable(var_type='_x', var_name='x_lon', shape=(np.shape(A_Alon)[0], 1))
        _u_lon = longitudinal_model.set_variable(var_type='_u', var_name='u_lon', shape=(np.shape(B_Blon)[1], 1))

        # Define state update in MPC Toolbox
        x_next_lon = A_Alon @ _x_lon + B_Blon @ _u_lon
        longitudinal_model.set_rhs('x_lon', x_next_lon)

        # Define cost function
        expression_lon = _x_lon.T @ Q_lon @ _x_lon
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
        }

        self.mpc_lon.set_param(**setup_longitudinal_mpc, nlpsol_opts=suppress_ipopt)

        # Setting up terminal cost I think?
        # mterm_lon = longitudinal_model.aux['cost']  # terminal cost
        mterm_lon = DM(1, 1)
        lterm_lon = longitudinal_model.aux['cost']  # terminal cost
        self.mpc_lon.set_objective(mterm=mterm_lon, lterm=lterm_lon)  # stage cost

        # This line is used in the toolbox
        self.mpc_lon.set_rterm(u_lon=R_lon)  # input penalty

        # Constraints
        max_u_lon = np.array([[np.radians(30)], [1.]])
        min_u_lon = np.array([[-np.radians(30)], [0.]])

        self.mpc_lon.bounds['upper', '_u', 'u_lon'] = max_u_lon
        self.mpc_lon.bounds['lower', '_u', 'u_lon'] = min_u_lon

        # Scaling
        scaling_array_lon = np.array([1, 1, 1, 1, 1])
        self.mpc_lon.scaling['_x', 'x_lon'] = scaling_array_lon

        # Setup the mpc
        self.mpc_lon.setup()

        # Initialize initial conditions
        self.mpc_lon.x0 = state.get_lon_state()
        self.mpc_lon.set_initial_guess()

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

        '''
        Longitudinal MPC
        '''
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

        lon_control = self.mpc_lon.make_step(x_lon)
        delta_e = lon_control[0, 0]
        delta_t = lon_control[1, 0]

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