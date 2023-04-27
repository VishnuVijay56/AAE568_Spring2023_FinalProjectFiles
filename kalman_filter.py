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
        self.var_lon_uvel_w  = 0.001
        self.var_lon_wvel_w  = 0.001
        self.var_lon_q_w     = 0.001
        self.var_lon_theta_w = 0.001
        self.var_lon_alt_w   = 0.001

        self.var_lon_uvel_v  = 0.04
        self.var_lon_wvel_v  = 0.06
        self.var_lon_q_v     = 0.0116
        self.var_lon_theta_v = 0.0014
        self.var_lon_alt_v   = 0.0021

        self.var_lat_V_w     = 0.001
        self.var_lat_p_w     = 0.001
        self.var_lat_r_w     = 0.001
        self.var_lat_phi_w   = 0.001
        self.var_lat_chi_w   = 0.001

        self.var_lat_V_v     = 0.07
        self.var_lat_p_v     = 0.0116
        self.var_lat_r_v     = 0.0014
        self.var_lat_phi_v   = 0.0021
        self.var_lat_chi_v   = 0.001

        self.P_lon = np.array([[self.var_lon_uvel_w**2, 0, 0, 0, 0], [0, self.var_lon_wvel_w**2, 0, 0, 0], [0, 0, self.var_lon_q_w**2, 0, 0], [0, 0, 0, self.var_lon_theta_w**2, 0], [0, 0, 0, 0, self.var_lon_alt_w**2]])
        self.P_lat = np.array([[self.var_lat_V_w**2, 0, 0, 0, 0], [0, self.var_lat_p_w**2, 0, 0, 0], [0, 0, self.var_lat_r_w**2, 0, 0], [0, 0, 0, self.var_lat_phi_w**2, 0], [0, 0, 0, 0, self.var_lat_chi_w**2]])

        self.q_lon = diags([self.var_lon_uvel_w**2, self.var_lon_wvel_w**2, self.var_lon_q_w**2, self.var_lon_theta_w**2, self.var_lon_alt_w**2])
        self.r_lon = diags([self.var_lon_uvel_v**2, self.var_lon_wvel_v**2, self.var_lon_q_v**2, self.var_lon_theta_v**2, self.var_lon_alt_v**2])

        self.q_lat = diags([self.var_lat_V_w**2, self.var_lat_p_w**2, self.var_lat_r_w**2, self.var_lat_phi_w**2, self.var_lat_chi_w**2])
        self.r_lat = diags([self.var_lat_V_v**2, self.var_lat_p_v**2, self.var_lat_r_v**2, self.var_lat_phi_v**2, self.var_lat_chi_v**2])

        #self.commanded_state = MAV_State()


    def update(self, states, delta):

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
        

        old_state = MAV_State()
        old_state.set_initial_cond(self.x_lon_hat_old, self.x_lat_hat_old)
        new_state = MAV_State()
        new_state.set_initial_cond(self.x_lon_hat_new, self.x_lat_hat_new)
        measured_state = MAV_State()
        measured_state.set_initial_cond(y_lon, y_lat)

        return old_state, measured_state

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
