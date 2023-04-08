"""
trim.py: find the trim state of the MAV
    - Author: Vishnu Vijay
    - Created: 6/25/22
"""

import numpy as np
from helper import EulerToQuaternion
from scipy.optimize import minimize
from delta_state import Delta_State
import mav_body_parameter as MAV
from mav_state import MAV_State


###
# Finds the trim state and trim inputs for desired flight velocity
# and flight path.
###
def compute_trim(mav_dynamics, Va, gamma):
    # initial conditions
    quat = EulerToQuaternion(0, gamma, 0)
    state0 = np.array([ [MAV.north0], #0
                        [MAV.east0], #1
                        [MAV.down0], #2
                        [MAV.u0], #3
                        [MAV.v0], #4
                        [MAV.w0], #5
                        [quat.item(0)], #6
                        [quat.item(1)], #7
                        [quat.item(2)], #8
                        [quat.item(3)], #9
                        [MAV.p0], #10
                        [MAV.q0], #11
                        [MAV.r0] #12
                        ])
    # print("initial state:\n", state0)
    delta0 = np.array([[mav_dynamics.delta.elevator_deflection],
                       [mav_dynamics.delta.aileron_deflection],
                       [mav_dynamics.delta.rudder_deflection],
                       [mav_dynamics.delta.throttle_level]])

    initial = np.concatenate((state0, delta0), axis=0)

    # define equality constraints
    cons = ({'type': 'eq',
             'fun': lambda x: np.array([
                                x[3]**2 + x[4]**2 + x[5]**2 - Va**2,  # magnitude of velocity vector is Va
                                x[4],  # v=0, force side velocity to be zero
                                x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 - 1.,  # force quaternion to be unit length
                                x[7],  # e1=0  - forcing e1=e3=0 ensures zero roll and zero yaw in trim
                                x[9],  # e3=0
                                x[10],  # p=0  - angular rates should all be zero
                                x[11],  # q=0
                                x[12],  # r=0
                                ]),
             'jac': lambda x: np.array([
                                [0., 0., 0., 2*x[3], 2*x[4], 2*x[5], 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 2*x[6], 2*x[7], 2*x[8], 2*x[9], 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                                ])
             })
    # solve the minimization problem to find the trim states and inputs

    res = minimize(trim_objective_fun, initial, method='SLSQP', args=(mav_dynamics, Va, gamma),
                   constraints=cons, options={'ftol': 1e-10, 'disp': False})
    # extract trim state and input and return
    trim_state = np.array([res.x[0:13]]).T
    trim_input = Delta_State()
    trim_input.elevator_deflection = res.x.item(13)
    trim_input.aileron_deflection = res.x.item(14)
    trim_input.rudder_deflection = res.x.item(15)
    trim_input.throttle_level = res.x.item(16)
    #trim_input.print()

    #print("trim state: ", trim_state)
    return trim_state, trim_input


def trim_objective_fun(x, mav_dynamics, Va, gamma):
    state = x[0:13]
    delta = Delta_State()
    delta.elevator_deflection = x.item(13)
    delta.aileron_deflection = x.item(14)
    delta.rudder_deflection = x.item(15)
    delta.throttle_level = x.item(16)
    
    desired_trim_state_dot = np.array([[0, 0, -Va*np.sin(gamma), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    mav_dynamics.internal_state = state
    mav_dynamics.update_velocity_data()

    forces_moments = mav_dynamics.net_force_moment(delta)
    f = mav_dynamics.derivatives(state, forces_moments)
    temp = desired_trim_state_dot - f
    J = np.linalg.norm(temp[2:13])**2
    return J