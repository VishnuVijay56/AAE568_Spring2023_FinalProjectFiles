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

class KalmanFilter:
    def __init__(self, ts_control):
        
        # x(k+1) = A*x(k) + B*u(k) + C*w(k)
        # y(k) = H*x(k) + G*v(k)

        # Noise & Sensor Matrices
        self.C_lat = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
        self.C_lon = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])

        self.H_lat = identity(5)
        self.H_lon = identity(5)

        self.G_lat = identity(5)
        self.G_lon = identity(5)

        #Initial Lon States
        self.x_lon_hat_old = MAV_State.get_lon_state()
        self.x_lon_hat_new = MAV_State.get_lon_state()
        self.x_lon_new = np.array([[0], 
                            [0], 
                            [0], 
                            [0],
                            [0]]) 
        #Initial Lat States
        self.x_lat_hat_old = MAV_State.get_lat_state()
        self.x_lat_hat_new = MAV_State.get_lat_state()
        self.x_lat_new = np.array([[0], 
                            [0], 
                            [0], 
                            [0],
                            [0]]) 
        # Noise & Error Covariance
        self.sig1_w = 100
        self.sig2_w = 1
        self.sig3_w = 100
        self.sig4_w = 1
        self.sig5_w = 1

        self.sig1_v = 1
        self.sig2_v = 1
        self.sig3_v = 1
        self.sig4_v = 1
        self.sig5_v = 1

        self.P_lon = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
        self.P_lat = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])

        self.q_lon = diags([self.sig1_w**2, self.sig2_w**2, self.sig3_w**2, self.sig4_w**2, self.sig5_w**2])
        self.r_lon = diags([self.sig1_v**2, self.sig2_v**2, self.sig3_v**2, self.sig4_v**2, self.sig5_v**2])

        self.q_lat = diags([self.sig1_w**2, self.sig2_w**2, self.sig3_w**2, self.sig4_w**2, self.sig5_w**2])
        self.r_lat = diags([self.sig1_v**2, self.sig2_v**2, self.sig3_v**2, self.sig4_v**2, self.sig5_v**2])

        #self.commanded_state = MAV_State()


    def update(self, cmd, state, delta):

        #############################
        # Longitudinal Kalman Filter
        #############################

        # Update State x{k}
        x_lon = MAV_State.get_lon_state()

        # Update the index of old and new
        self.x_lon_hat_old = self.x_lon_hat_new

        # Get Output y{k} = H*x{k} + G*v{k}
        y_lon = self.H_lon @ x_lon + self.G_lon @ self.getSensorNoise()

        # x{k+1} = A * x{k} + B*u{k} + C*w{k}
        self.x_lon_new = M.Ad_lon @ x_lon + M.Bd_lon @ delta.get_ulon() + self.C_lon @ self.getProcessNoise()
        
        # Smoothing Equations for xhat{k}
        self.x_lon_hat_old = self.x_lon_hat_old + self.P_lon @ np.transpose(self.H_lon) @ pinv(self.H_lon @ self.P_lon @ np.transpose(self.H_lon) + self.G_lon @ self.r_lon @ np.transpose(self.G_lon)) @ (y_lon - self.H_lon @ self.x_lon_hat_old)
        
        # Kalman Filter Gain
        L_lon = M.Ad_lon @ self.P_lon @ np.transpose(self.H_lon) @ pinv(self.H_lon @ self.P_lon @ np.transpose(self.H_lon) + self.G_lon @ self.r_lon @ np.transpose(self.G_lon))
        
        # xhat{k+1}
        self.x_lon_hat_new = (M.Ad_lon - L_lon @ self.H_lon) @ self.x_lon_hat_old + L_lon @ y_lon + M.Bd_lon @ delta.get_ulon()
        
        # Update Error Covariance
        self.P_lon = (M.Ad_lon - L_lon @ self.H_lon) @ self.P_lon @ np.transpose(M.Ad_lon - L_lon @ self.H_lon) + self.C_lon @ self.q_lon @ np.transpose(self.C_lon) + L_lon @ self.G_lon @ self.r_lon @ np.transpose(self.G_lon) @ np.transpose(L_lon)

        #############################
        # Latitudinal Kalman Filter
        #############################

        # Update State x{k}
        x_lat = MAV_State.get_lat_state()

        # Update the index of old and new
        self.x_lat_hat_old = self.x_lat_hat_new

        # Get Output y{k} = H*x{k} + G*v{k}
        y_lat = self.H_lat @ x_lat + self.G_lat @ self.getSensorNoise()

        # x{k+1} = A * x{k} + B*u{k} + C*w{k}
        self.x_lat_new = M.Ad_lat @ x_lat + M.Bd_lat @ delta.get_ulat() + self.C_lat @ self.getProcessNoise()
        
        # Smoothing Equations for xhat{k}
        self.x_lat_hat_old = self.x_lat_hat_old + self.P_lat @ np.transpose(self.H_lat) @ pinv(self.H_lat @ self.P_lat @ np.transpose(self.H_lat) + self.G_lat @ self.r_lat @ np.transpose(self.G_lat)) @ (y_lat - self.H_lat @ self.x_lat_hat_old)
        
        # Kalman Filter Gain
        L_lat = M.Ad_lat @ self.P_lat @ np.transpose(self.H_lat) @ pinv(self.H_lat @ self.P_lat @ np.transpose(self.H_lat) + self.G_lat @ self.r_lat @ np.transpose(self.G_lat))
        
        # xhat{k+1}
        self.x_lat_hat_new = (M.Ad_lat - L_lat @ self.H_lat) @ self.x_lat_hat_old + L_lat @ y_lat + M.Bd_lat @ delta.get_ulat()
        
        # Update Error Covariance
        self.P_lat = (M.Ad_lat - L_lat @ self.H_lat) @ self.P_lat @ np.transpose(M.Ad_lat - L_lat @ self.H_lat) + self.C_lat @ self.q_lat @ np.transpose(self.C_lat) + L_lat @ self.G_lat @ self.r_lat @ np.transpose(self.G_lat) @ np.transpose(L_lat)

        #############################
        # Lidar & Radar Kalman Filter
        #############################


        return self.x_lon_hat_old, self.x_lon_hat_new, self.x_lat_hat_old, self.x_lat_hat_new

    def getProcessNoise(self):
        w = np.array([[self.sig1_w**2 *np.random.randn()], 
                 [self.sig2_w**2 *np.random.randn()], 
                 [self.sig3_w**2 *np.random.randn()],
                 [self.sig4_w**2 *np.random.randn()],
                 [self.sig5_w**2 *np.random.randn()]])
        return w
    
    def getSensorNoise(self):
        v = np.array([[self.sig1_v**2 *np.random.randn()], 
                    [self.sig2_v**2 *np.random.randn()], 
                    [self.sig3_v**2 *np.random.randn()],
                    [self.sig4_v**2 *np.random.randn()],
                    [self.sig5_v**2 *np.random.randn()]])
        return v