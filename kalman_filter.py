"""
kalman_filter.py: kalman filter to estimate states from noisy data
    - Author: Nathan Berry
    - Created: 4/08/23
"""

import numpy as np
import model_coef as TF
import mav_body_parameter as MAV
from mav_state import MAV_State
from delta_state import Delta_State
from scipy.linalg import expm, pinv, eigvals
import control_parameters as AP
import model_coef_discrete as M
from scipy.sparse import identity, diags
from scipy.sparse.linalg import eigs
import model_coef as MC
import mav_body_parameter as MAV_para
import math

class KalmanFilter:
    def __init__(self, states:MAV_State):
        
        # x(k+1) = A*x(k) + B*u(k) + C*w(k)
        # y(k) = H*x(k) + G*v(k)

        # Noise & Sensor Matrices
        self.C_lat = identity(5)
        self.C_lon = identity(5)

        self.H_lat = identity(5)
        self.H_lon = identity(5)

        self.G_lat = identity(5)
        self.G_lon = identity(5)

        #Initial Lon States
        self.x_lon_hat_old = states.get_lon_state()
        self.x_lon_hat_new = states.get_lon_state()
        #self.x_lon_new = np.array([[0], [0], [0], [0], [0]]) 
        #Initial Lat States
        self.x_lat_hat_old = states.get_lat_state()
        self.x_lat_hat_new = states.get_lat_state()
        self.x_lat_new = np.array([[0], 
                            [0], 
                            [0], 
                            [0],
                            [0]]) 
        
        # Noise & Error Covariance
        self.var_lon_uvel_w  = 1
        self.var_lon_wvel_w  = 1
        self.var_lon_q_w     = 0.01
        self.var_lon_theta_w = 0.01
        self.var_lon_alt_w   = 0.01

        self.var_lon_uvel_v  = 0.01
        self.var_lon_wvel_v  = 0.01
        self.var_lon_q_v     = 0.01
        self.var_lon_theta_v = 0.01
        self.var_lon_alt_v   = 0.01

        self.var_lat_V_w     = 0.01
        self.var_lat_p_w     = 0.01
        self.var_lat_r_w     = 0.01
        self.var_lat_phi_w   = 0.01
        self.var_lat_chi_w   = 0.01

        self.var_lat_V_v     = 0.01
        self.var_lat_p_v     = 0.01
        self.var_lat_r_v     = 0.01
        self.var_lat_phi_v   = 0.01
        self.var_lat_chi_v   = 0.01

        self.P_lon = np.array([[self.var_lon_uvel_w, 0, 0, 0, 0], [0, self.var_lon_wvel_w, 0, 0, 0], [0, 0, self.var_lon_q_w, 0, 0], [0, 0, 0, self.var_lon_theta_w, 0], [0, 0, 0, 0, self.var_lon_alt_w]])
        self.P_lat = np.array([[self.var_lat_V_w, 0, 0, 0, 0], [0, self.var_lat_p_w, 0, 0, 0], [0, 0, self.var_lat_r_w, 0, 0], [0, 0, 0, self.var_lat_phi_w, 0], [0, 0, 0, 0, self.var_lat_chi_w]])

        self.q_lon = diags([self.var_lon_uvel_w, self.var_lon_wvel_w, self.var_lon_q_w, self.var_lon_theta_w, self.var_lon_alt_w])
        self.r_lon = diags([self.var_lon_uvel_v, self.var_lon_wvel_v, self.var_lon_q_v, self.var_lon_theta_v, self.var_lon_alt_v])

        self.q_lat = diags([self.var_lat_V_w, self.var_lat_p_w, self.var_lat_r_w, self.var_lat_phi_w, self.var_lat_chi_w])
        self.r_lat = diags([self.var_lat_V_v, self.var_lat_p_v, self.var_lat_r_v, self.var_lat_phi_v, self.var_lat_chi_v])

        #self.commanded_state = MAV_State()


    def update(self, estimated_states, states, delta):

        #############################
        # Longitudinal Kalman Filter
        #############################
        # Update State x{k}
        x_lon = states.get_lon_state()

        # Update the index of old and new
        self.x_lon_hat_old = self.x_lon_hat_new

        # Get Output y{k} = H*x{k} + G*v{k}
        y_lon = self.H_lon @ x_lon + self.G_lon @ self.getSensorNoise_lon()

        # x{k+1} = A * x{k} + B*u{k} + C*w{k}
        #self.x_lon_new = M.Ad_lon @ x_lon + M.Bd_lon @ delta.get_ulon() + self.C_lon @ self.getProcessNoise_lon()
        
        # Smoothing Equations for xhat{k}
        self.x_lon_hat_old = self.x_lon_hat_old + self.P_lon @ np.transpose(self.H_lon) @ pinv(self.H_lon @ self.P_lon @ np.transpose(self.H_lon) + self.G_lon @ self.r_lon @ np.transpose(self.G_lon)) @ (y_lon - self.H_lon @ self.x_lon_hat_old)
        
        # Kalman Filter Gain
        L_lon = M.Ad_lon @ self.P_lon @ np.transpose(self.H_lon) @ pinv(self.H_lon @ self.P_lon @ np.transpose(self.H_lon) + self.G_lon @ self.r_lon @ np.transpose(self.G_lon))
        
        # xhat{k+1}
        self.x_lon_hat_new = (M.Ad_lon - L_lon @ self.H_lon) @ self.x_lon_hat_old + L_lon @ y_lon + M.Bd_lon @ delta.get_ulon()
        
        # Update Error Covariance
        #self.P_lon = (M.Ad_lon - L_lon @ self.H_lon) @ self.P_lon @ np.transpose(M.Ad_lon - L_lon @ self.H_lon) + self.C_lon @ self.q_lon @ np.transpose(self.C_lon) + L_lon @ self.G_lon @ self.r_lon @ np.transpose(self.G_lon) @ np.transpose(L_lon)
        self.P_lon = M.Ad_lon @ self.P_lon @ np.transpose(M.Ad_lon) + self.C_lon @ self.q_lon @ np.transpose(self.C_lon) - (M.Ad_lon @ self.P_lon @ np.transpose(self.H_lon)) @ pinv(self.H_lon @ self.P_lon @ np.transpose(self.H_lon) + self.G_lon @ self.r_lon @ np.transpose(self.G_lon)) @ (self.H_lon @ self.P_lon @ np.transpose(M.Ad_lon))

        #############################
        # Latitudinal Kalman Filter
        #############################
        
        # Update State x{k}
        x_lat = states.get_lat_state()

        # Update the index of old and new
        self.x_lat_hat_old = self.x_lat_hat_new

        # Get Output y{k} = H*x{k} + G*v{k}
        y_lat = self.H_lat @ x_lat + self.G_lat @ self.getSensorNoise_lat()

        # x{k+1} = A * x{k} + B*u{k} + C*w{k}
        #self.x_lat_new = M.Ad_lat @ x_lat + M.Bd_lat @ delta.get_ulat() + self.C_lat @ self.getProcessNoise_lat()
        self.x_lat_new = M.Ad_lat @ x_lat + M.Bd_lat @ delta.get_ulat() + self.C_lat @ self.getProcessNoise_lat()

        # Smoothing Equations for xhat{k}
        self.x_lat_hat_old = self.x_lat_hat_old + self.P_lat @ np.transpose(self.H_lat) @ pinv(self.H_lat @ self.P_lat @ np.transpose(self.H_lat) + self.G_lat @ self.r_lat @ np.transpose(self.G_lat)) @ (y_lat - self.H_lat @ self.x_lat_hat_old)
        
        # Kalman Filter Gain
        #L_lat = M.Ad_lat @ self.P_lat @ np.transpose(self.H_lat) @ pinv(self.H_lat @ self.P_lat @ np.transpose(self.H_lat) + self.G_lat @ self.r_lat @ np.transpose(self.G_lat))
        L_lat = M.Ad_lat @ self.P_lat @ np.transpose(self.H_lat) @ pinv(self.H_lat @ self.P_lat @ np.transpose(self.H_lat) + self.G_lat @ self.r_lat @ np.transpose(self.G_lat))
        # xhat{k+1}
        #self.x_lat_hat_new = (M.Ad_lat - L_lat @ self.H_lat) @ self.x_lat_hat_old + L_lat @ y_lat + M.Bd_lat @ delta.get_ulat()
        self.x_lat_hat_new = (M.Ad_lat - L_lat @ self.H_lat) @ self.x_lat_hat_old + L_lat @ y_lat + M.Bd_lat @ delta.get_ulat()
        # Update Error Covariance
        #self.P_lat = (M.Ad_lat - L_lat @ self.H_lat) @ self.P_lat @ np.transpose(M.Ad_lat - L_lat @ self.H_lat) + self.C_lat @ self.q_lat @ np.transpose(self.C_lat) + L_lat @ self.G_lat @ self.r_lat @ np.transpose(self.G_lat) @ np.transpose(L_lat)
        #self.P_lat = (M.Ad_lat - L_lat @ self.H_lat) @ self.P_lat @ np.transpose(M.Ad_lat - L_lat @ self.H_lat) + self.C_lat @ self.q_lat @ np.transpose(self.C_lat) + L_lat @ self.G_lat @ self.r_lat @ np.transpose(self.G_lat) @ np.transpose(L_lat)
        self.P_lat = M.Ad_lat @ self.P_lat @ np.transpose(M.Ad_lat) + self.C_lat @ self.q_lat @ np.transpose(self.C_lat) - (M.Ad_lat @ self.P_lat @ np.transpose(self.H_lat)) @ pinv(self.H_lat @ self.P_lat @ np.transpose(self.H_lat) + self.G_lat @ self.r_lat @ np.transpose(self.G_lat)) @ (self.H_lat @ self.P_lat @ np.transpose(M.Ad_lat))
        
        #############################
        # Lidar & Radar Kalman Filter
        #############################

        old_state = MAV_State()
        old_state.set_initial_cond(self.x_lon_hat_old, self.x_lat_hat_old)
        new_state = MAV_State()
        new_state.set_initial_cond(self.x_lon_hat_new, self.x_lat_hat_new)

        return old_state

    def getProcessNoise_lon(self):
        w = np.array([[self.var_lon_uvel_w *np.random.randn()], 
                 [self.var_lon_wvel_w *np.random.randn()], 
                 [self.var_lon_q_w *np.random.randn()],
                 [self.var_lon_theta_w *np.random.randn()],
                 [self.var_lon_alt_w *np.random.randn()]])
        return w

    def getSensorNoise_lon(self):
        v = np.array([[self.var_lon_uvel_v *np.random.randn()], 
                    [self.var_lon_wvel_v *np.random.randn()], 
                    [self.var_lon_q_v *np.random.randn()],
                    [self.var_lon_theta_v *np.random.randn()],
                    [self.var_lon_alt_v *np.random.randn()]])
        return v

    def getProcessNoise_lat(self):
        w = np.array([[self.var_lat_V_w *np.random.randn()], 
                 [self.var_lat_p_w *np.random.randn()], 
                 [self.var_lat_r_w *np.random.randn()],
                 [self.var_lat_phi_w *np.random.randn()],
                 [self.var_lat_chi_w *np.random.randn()]])
        return w

    def getSensorNoise_lat(self):
        v = np.array([[self.var_lat_V_v *np.random.randn()], 
                    [self.var_lat_p_v *np.random.randn()], 
                    [self.var_lat_r_v *np.random.randn()],
                    [self.var_lat_phi_v *np.random.randn()],
                    [self.var_lat_chi_v *np.random.randn()]])
        return v
"""
def jacobian_lat(x_lat, x_lon, u_lat):
    Ts = 0.010
    v = x_lat[0]
    p = x_lat[1]
    r = x_lat[2]
    phi = x_lat[3]
    chi = x_lat[4]
    u = x_lon[0]
    w = x_lon[1]
    q = x_lon[2]
    theta = x_lon[3]
    h = x_lon[4]
    beta = math.atan(v/math.sqrt(u**2 + w**2))
    Va = math.sqrt(u**2 + v**2 + w**2)
    
    Cp0 = MAV_para.gamma3 * MAV_para.C_l_0 + MAV_para.gamma4 * MAV_para.C_n_0
    Cpb = MAV_para.gamma3 * MAV_para.C_l_beta + MAV_para.gamma4 * MAV_para.C_n_beta
    Cpdela = MAV_para.gamma3 * MAV_para.C_l_delta_a + MAV_para.gamma4 * MAV_para.C_n_delta_a
    Cpdelr = MAV_para.gamma3 * MAV_para.C_l_delta_r + MAV_para.gamma4 * MAV_para.C_n_delta_r
    Cpp = MAV_para.gamma3 * MAV_para.C_l_p + MAV_para.gamma4 * MAV_para.C_n_p
    Cpr = MAV_para.gamma3 * MAV_para.C_l_r + MAV_para.gamma4 * MAV_para.C_n_r
    Crp = MAV_para.gamma4 * MAV_para.C_l_p + MAV_para.gamma8 * MAV_para.C_n_p
    Crr = MAV_para.gamma4 * MAV_para.C_l_r + MAV_para.gamma8 * MAV_para.C_n_r
    Cr0 = MAV_para.gamma4 * MAV_para.C_l_0 + MAV_para.gamma8 * MAV_para.C_n_0
    Crb = MAV_para.gamma4 * MAV_para.C_l_beta + MAV_para.gamma8 * MAV_para.C_n_beta
    Crdela = MAV_para.gamma4 * MAV_para.C_l_delta_a + MAV_para.gamma8 * MAV_para.C_n_delta_a
    Crdelr = MAV_para.gamma4 * MAV_para.C_l_delta_r + MAV_para.gamma8 * MAV_para.C_n_delta_r

    Yv = (MAV_para.rho * MAV_para.S_wing * MAV_para.b * v)/(4 * MAV_para.m * Va) * (MAV_para.C_Y_p * p + MAV_para.C_Y_r * r)  \
         + (MAV_para.rho * MAV_para.S_wing * v)/(MAV_para.m) * (MAV_para.C_Y_0 + MAV_para.C_Y_beta * beta + MAV_para.C_Y_delta_a * u_lat[0] + MAV_para.C_Y_delta_r * u_lat[1]) \
         + (MAV_para.rho * MAV_para.S_wing * MAV_para.C_Y_beta)/(2 * MAV_para.m) * math.sqrt(u**2 + w**2)
    Yp = w + (MAV_para.rho * Va * MAV_para.S_wing * MAV_para.b)/(4 * MAV_para.m) * MAV_para.C_Y_p
    Yr = -u + (MAV_para.rho * Va * MAV_para.S_wing * MAV_para.b)/(4 * MAV_para.m) * MAV_para.C_Y_r
    Lv = (MAV_para.rho * MAV_para.S_wing * MAV_para.b**2 * v)/(4 * Va) * (Cp0 + Cpb * beta + Cpdela * u_lat[0] + Cpdelr * u_lat[1]) + (MAV_para.rho * MAV_para.S_wing * MAV_para.b * Cpb)/(2) * math.sqrt(u**2 + w**2)
    Lp = MAV_para.gamma1 * q + (MAV_para.rho * Va * MAV_para.S_wing * MAV_para.b**2)/(4) * Cpp
    Lr = -MAV_para.gamma2 * q + (MAV_para.rho * Va * MAV_para.S_wing * MAV_para.b**2)/(4) * Cpr
    Nv = (MAV_para.rho * MAV_para.S_wing * MAV_para.b**2 * v)/(4 * Va) * (Crp * p + Crr * r) + (MAV_para.rho * MAV_para.S_wing * MAV_para.b * v) * (Cr0 + Crb*beta + Crdela * u_lat[0] + Crdelr*u_lat[1]) + (MAV_para.rho * MAV_para.S_wing * MAV_para.b * Crb)/(2) * math.sqrt(u**2 + w**2)                                                                                    
    Np = MAV_para.gamma7 * q + (MAV_para.rho * Va * MAV_para.S_wing * MAV_para.b**2)/(4) * Crp
    Nr = -MAV_para.gamma1 * q + (MAV_para.rho * Va * MAV_para.S_wing * MAV_para.b**2)/(4) * Crr

    A14 = MAV_para.gravity * math.cos(theta) * math.cos(phi)
    A43 = math.cos(phi) * math.tan(theta)
    A44 = q * math.cos(phi)*math.tan(theta) - r*math.sin(phi)*math.tan(theta)
    A53 = math.cos(phi) * (1/math.cos(theta))
    A54 = p * math.cos(phi) * (1/math.cos(theta)) - r*math.sin(phi)*(1/math.cos(theta))

    #Save in Matrix
    A_lat = np.array([[Yv, Yp, Yr, A14, 0], 
                      [Lv, Lp, Lr, 0, 0],
                      [Nv, Np, Nr, 0, 0], 
                      [0, 1, A43, A44, 0],
                      [0, 0, A53, A54, 0]], dtype=object) #row 5 change from phi [0, 0, A53, A54, 0] to linear chi [0.000000, -0.000002, 1.001223, 0.000000, 0.000000]]
    # Discretize
    A_lat = expm(A_lat*Ts)

    #########################
    # B_lat
    #########################
    Ndela = (MAV_para.rho * Va**2 * MAV_para.S_wing * MAV_para.b)/(2) * Crdela
    Ndelr = (MAV_para.rho * Va**2 * MAV_para.S_wing * MAV_para.b)/(2) * Crdelr
    Ldela = (MAV_para.rho * Va**2 * MAV_para.S_wing * MAV_para.b)/(2) * Cpdela
    Ldelr = (MAV_para.rho * Va**2 * MAV_para.S_wing * MAV_para.b)/(2) * Cpdelr
    Ydela = (MAV_para.rho * Va**2 * MAV_para.S_wing)/(2  * MAV_para.m) * MAV_para.C_Y_delta_a
    Ydelr =  (MAV_para.rho * Va**2 * MAV_para.S_wing)/(2  * MAV_para.m) * MAV_para.C_Y_delta_r

    B_lat = np.array([[Ydela, Ydelr], 
                      [Ldela, Ldelr],
                      [Ndela, Ndelr],
                      [0, 0],
                      [0, 0]])
    
    B_lat = pinv(A_lat) @ (A_lat-np.identity(5)) @ B_lat

    return A_lat, B_lat
"""