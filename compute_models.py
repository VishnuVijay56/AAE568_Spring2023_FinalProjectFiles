"""
compute_models.py: Computes the transfer functions and state space models described in chapter 5
    - Name: Vishnu Vijay
    - Date Created: 6/26/22
"""

import numpy as np
from scipy.optimize import minimize
from helper import EulerToQuaternion, QuaternionToEuler
import mav_body_parameter as MAV
from delta_state import Delta_State


def compute_model(mav, trim_state, trim_input):
    Ts = 0.01
    A_lon, B_lon, A_lat, B_lat = compute_ss_model(mav, trim_state, trim_input)
    Va_trim, alpha_trim, theta_trim, a_phi1, a_phi2, a_theta1, a_theta2, a_theta3, \
    a_V1, a_V2, a_V3 = compute_tf_model(mav, trim_state, trim_input)

    # write transfer function gains to file
    file = open('model_coef.py', 'w')
    file.write('import numpy as np\n')
    file.write('x_trim = np.array([[%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f]]).T\n' %
               (trim_state.item(0), trim_state.item(1), trim_state.item(2), trim_state.item(3),
                trim_state.item(4), trim_state.item(5), trim_state.item(6), trim_state.item(7),
                trim_state.item(8), trim_state.item(9), trim_state.item(10), trim_state.item(11),
                trim_state.item(12)))
    file.write('u_trim = np.array([[%f, %f, %f, %f]]).T\n' %
               (trim_input.elevator_deflection, trim_input.aileron_deflection, trim_input.rudder_deflection, trim_input.throttle_level))
    file.write('Va_trim = %f\n' % Va_trim)
    file.write('alpha_trim = %f\n' % alpha_trim)
    file.write('theta_trim = %f\n' % theta_trim)
    file.write('a_phi1 = %f\n' % a_phi1)
    file.write('a_phi2 = %f\n' % a_phi2)
    file.write('a_theta1 = %f\n' % a_theta1)
    file.write('a_theta2 = %f\n' % a_theta2)
    file.write('a_theta3 = %f\n' % a_theta3)
    file.write('a_V1 = %f\n' % a_V1)
    file.write('a_V2 = %f\n' % a_V2)
    file.write('a_V3 = %f\n' % a_V3)
    file.write('A_lon = np.array([\n    [%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f]])\n' %
    (A_lon[0][0], A_lon[0][1], A_lon[0][2], A_lon[0][3], A_lon[0][4],
     A_lon[1][0], A_lon[1][1], A_lon[1][2], A_lon[1][3], A_lon[1][4],
     A_lon[2][0], A_lon[2][1], A_lon[2][2], A_lon[2][3], A_lon[2][4],
     A_lon[3][0], A_lon[3][1], A_lon[3][2], A_lon[3][3], A_lon[3][4],
     A_lon[4][0], A_lon[4][1], A_lon[4][2], A_lon[4][3], A_lon[4][4]))
    file.write('B_lon = np.array([\n    [%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f]])\n' %
    (B_lon[0][0], B_lon[0][1],
     B_lon[1][0], B_lon[1][1],
     B_lon[2][0], B_lon[2][1],
     B_lon[3][0], B_lon[3][1],
     B_lon[4][0], B_lon[4][1],))
    file.write('A_lat = np.array([\n    [%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f]])\n' %
    (A_lat[0][0], A_lat[0][1], A_lat[0][2], A_lat[0][3], A_lat[0][4],
     A_lat[1][0], A_lat[1][1], A_lat[1][2], A_lat[1][3], A_lat[1][4],
     A_lat[2][0], A_lat[2][1], A_lat[2][2], A_lat[2][3], A_lat[2][4],
     A_lat[3][0], A_lat[3][1], A_lat[3][2], A_lat[3][3], A_lat[3][4],
     A_lat[4][0], A_lat[4][1], A_lat[4][2], A_lat[4][3], A_lat[4][4]))
    file.write('B_lat = np.array([\n    [%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f]])\n' %
    (B_lat[0][0], B_lat[0][1],
     B_lat[1][0], B_lat[1][1],
     B_lat[2][0], B_lat[2][1],
     B_lat[3][0], B_lat[3][1],
     B_lat[4][0], B_lat[4][1],))
    file.write('Ts = %f\n' % Ts)
    file.close()


def compute_tf_model(mav, trim_state, trim_input):
    # trim values
    mav.internal_state = trim_state
    mav.update_velocity_data()
    Va_trim = mav.V_a
    alpha_trim = mav.alpha
    phi, theta_trim, psi = QuaternionToEuler(trim_state[6:10])

    # define transfer function constants
    a_phi1 = -0.25*MAV.rho*(mav.V_a)*MAV.S_wing*(MAV.b**2)*(MAV.gamma3*MAV.C_l_p + MAV.gamma4*MAV.C_n_p)
    a_phi2 = 0.5*MAV.rho*(mav.V_a**2)*MAV.S_wing*MAV.b*(MAV.gamma3*MAV.C_l_delta_a + MAV.gamma4*MAV.C_n_delta_a)
    a_theta1 = -0.25*MAV.rho*(mav.V_a**2)*MAV.c*MAV.S_wing*MAV.C_m_q*MAV.c/(MAV.Jy*mav.V_a)
    a_theta2 = -0.5*MAV.rho*(mav.V_a**2)*MAV.c*MAV.S_wing*MAV.C_m_alpha/(MAV.Jy)
    a_theta3 = 0.5*MAV.rho*(mav.V_a**2)*MAV.c*MAV.S_wing*MAV.C_m_delta_e/(MAV.Jy)

    # Compute transfer function coefficients using new propulsion model
    a_V1 = MAV.rho*Va_trim*MAV.S_wing*(MAV.C_D_0 + MAV.C_D_alpha*alpha_trim + MAV.C_D_delta_e*trim_input.elevator_deflection)
    a_V1 = (a_V1 - dT_dVa(mav, Va_trim, trim_input.throttle_level)) / MAV.m
    a_V2 = dT_ddelta_t(mav, Va_trim, trim_input.throttle_level) / MAV.m
    a_V3 = MAV.gravity * np.cos(theta_trim - alpha_trim)

    return Va_trim, alpha_trim, theta_trim, a_phi1, a_phi2, a_theta1, a_theta2, a_theta3, a_V1, a_V2, a_V3


def compute_ss_model(mav, trim_state, trim_input):
    # Find A and B for the longitudinal and lateral state spaces
    A_lon, B_lon, A_lat, B_lat = find_ss_matrices(mav, trim_state, trim_input)
    
    ## Eigenval calculation
    
    import pandas as pd
    pd.set_option('display.width',500)
    pd.set_option('display.max_columns',12)
    np.set_printoptions(linewidth=500)
    
    #A_lon_eigval, A_lon_eigvec = np.linalg.eig(A_lon)
    #A_lat_eigval, A_lat_eigvec = np.linalg.eig(A_lat)

    #print("\nA_lon_eig: ", end="  ")
    #for i in range(A_lon_eigval.size):
    #    print(A_lon_eigval[i], end="   ")
    #print()
    
    #print("\nA_lat_eig: ", end=" ")
    #for i in range(A_lat_eigval.size):
    #    print(A_lat_eigval[i], end="   ")
    #print()

    return A_lon, B_lon, A_lat, B_lat


def euler_state(x_quat):
    # convert state x with attitude represented by quaternion
    # to x_euler with attitude represented by Euler angles
    phi, theta, psi = QuaternionToEuler(x_quat[6:10])
    x_euler = np.empty((12, 1))
    x_euler[0] = [x_quat.item(0)]
    x_euler[1] = [x_quat.item(1)]
    x_euler[2] = [x_quat.item(2)]
    x_euler[3] = [x_quat.item(3)]
    x_euler[4] = [x_quat.item(4)]
    x_euler[5] = [x_quat.item(5)]
    x_euler[6] = [phi]
    x_euler[7] = [theta]
    x_euler[8] = [psi]
    x_euler[9] = [x_quat.item(10)]
    x_euler[10] = [x_quat.item(11)]
    x_euler[11] = [x_quat.item(12)]
    return x_euler


def quaternion_state(x_euler):
    # convert state x_euler with attitude represented by Euler angles
    # to x_quat with attitude represented by quaternions
    e_quat = EulerToQuaternion(x_euler.item(6), x_euler.item(7), x_euler.item(8))
    x_quat = np.array([[x_euler.item(0)],
                        [x_euler.item(1)],
                        [x_euler.item(2)],
                        [x_euler.item(3)],
                        [x_euler.item(4)],
                        [x_euler.item(5)],
                        [e_quat.item(0)],
                        [e_quat.item(1)],
                        [e_quat.item(2)],
                        [e_quat.item(3)],
                        [x_euler.item(9)],
                        [x_euler.item(10)],
                        [x_euler.item(11)]])
    
    return x_quat


def f_quat(mav, x_quat, delta):
    # Find x_dot in quaternion state space
    mav.internal_state = x_quat
    mav.update_velocity_data(np.zeros((6,1)))
    f_m = mav.net_force_moment(delta)
    f_quat = mav.derivatives(x_quat, f_m)

    return f_quat


def dt_dq(mav, x_euler, delta):
    eps = 0.01
    dt_dq = np.zeros([12, 13])
    dt_dq[0:6, 0:6] = np.eye(6)
    dt_dq[9:, 10:] = np.eye(3)
    phi, theta, psi = QuaternionToEuler(x_euler[6:10])
    f = np.array([phi, theta, psi])
    for i in range(4):
        x_eps = np.copy(x_euler[6:10])
        x_eps[i][0] += eps

        new_phi, new_theta, new_psi = QuaternionToEuler(x_eps)
        f_eps = np.array([new_phi, new_theta, new_psi])
        df = (f_eps - f) / eps
        # print("dt_dq: ", df.T)
        for j in range(3):
            dt_dq[6+j][6+i] = df[j]
    # print()
    return dt_dq


def dtI_dq(mav, x_euler, delta):
    eps = 0.01
    [phi, theta, psi] = QuaternionToEuler(x_euler[6:10])
    dtI_dq = np.zeros([13, 12])
    dtI_dq[0:6, 0:6] = np.eye(6)
    dtI_dq[10:, 9:] = np.eye(3)
    f = EulerToQuaternion(phi, theta, psi)
    for i in range(3):
        x_eps = np.array([phi, theta, psi])
        x_eps[i] += eps
        new_phi = x_eps.item(0)
        new_theta = x_eps.item(1)
        new_psi = x_eps.item(2)
        f_eps = EulerToQuaternion(new_phi, new_theta, new_psi)
        # print("f_eps:\n", f_eps)
        # print("f.T:\n", f.T[0])
        df = (f_eps - f.T[0]) / eps
        # print("dtI_dq: ", df.T)
        for j in range(4):
            dtI_dq[6+j][6+i] = df[j]
    # print()
    return dtI_dq


def df_dx(mav, x_quat, delta):
    # take partial of f_euler with respect to x_euler
    eps = 0.01
    m = 12 + 1
    n = 12 + 1
    A = np.zeros((m, n))

    f_at_x = f_quat(mav, x_quat, delta)
    for i in range(n):
        x_eps = np.copy(x_quat)
        x_eps[i][0] += eps # add eps to ith state
        
        f_at_x_eps = f_quat(mav, x_eps, delta)
        df_dxi = (f_at_x_eps - f_at_x) / eps
        
        A[:, i] = df_dxi[:, 0]
    
    return A


def df_du(mav, x_euler, delta):
    # take partial of f_euler with respect to input
    eps = 0.01
    m = 12 + 1
    n = 4
    B = np.zeros((m, n))

    f_at_x = f_quat(mav, x_euler, delta)
    for i in range(n):
        #print("i: ", i, end=" ")
        delta_eps_arr = np.array([[delta.elevator_deflection],
                              [delta.aileron_deflection],
                              [delta.rudder_deflection],
                              [delta.throttle_level]])
        delta_eps_arr[i][0] += eps # add eps to ith state
        
        delta_eps = Delta_State()
        delta_eps.elevator_deflection = delta_eps_arr[0]
        delta_eps.aileron_deflection = delta_eps_arr[1]
        delta_eps.rudder_deflection = delta_eps_arr[2]
        delta_eps.throttle_level = delta_eps_arr[3]

        f_at_x_eps = f_quat(mav, x_euler, delta_eps)
        df_dui = (f_at_x_eps - f_at_x) / eps
        B[:, i] = df_dui[:, 0]

    return B


def find_ss_matrices(mav, x_quat, delta):
    #print("TRIM_STATE:\n", x_quat)
    Aq = df_dx(mav, x_quat, delta)
    Bq = df_du(mav, x_quat, delta)

    T = dt_dq(mav, x_quat, delta)
    T_inv = dtI_dq(mav, x_quat, delta)
    
    # Convert from quat state space to euler state space
    A = T @ Aq @ T_inv
    B = T @ Bq
    # Zero height state
    A[2,:]=-A[2,:]
    B[2,:]=-B[2,:]

    # extract longitudinal states (u, w, q, theta, pd) and change pd to h
    E1 = np.zeros((5, 12))
    E1[0][3] = 1
    E1[1][5] = 1
    E1[2][10] = 1
    E1[3][7] = 1
    E1[4][2] = -1
    E2 = np.zeros((2, 4))
    E2[0][0] = 1
    E2[1][3] = 1
    A_lon = E1 @ A @ E1.T
    B_lon = E1 @ B @ E2.T
    # extract lateral states (v, p, r, phi, psi)
    E3 = np.zeros((5, 12))
    E3[0][4] = 1
    E3[1][9] = 1
    E3[2][11] = 1
    E3[3][6] = 1
    E3[4][8] = 1
    E4 = np.zeros((2, 4))
    E4[0][1] = 1
    E4[1][2] = 1
    A_lat = E3 @ A @ (E3.T)
    B_lat = E3 @ B @ (E4.T)

    return A_lon, B_lon, A_lat, B_lat


def dT_dVa(mav, Va, delta_t):
    # returns the derivative of motor thrust with respect to Va
    eps = 0.001
    T_eps, Q_eps = mav.motor_thrust_torque(Va + eps, delta_t)
    T, Q = mav.motor_thrust_torque(Va, delta_t)
    return (T_eps - T) / eps


def dT_ddelta_t(mav, Va, delta_t):
    # returns the derivative of motor thrust with respect to delta_t
    eps = 0.001
    T_eps, Q_eps = mav.motor_thrust_torque(Va, delta_t + eps)
    T, Q = mav.motor_thrust_torque(Va, delta_t)
    return (T_eps - T) / eps