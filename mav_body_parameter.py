"""
mav_body_parameters.py: collection of the MAV's body parameters
    - Author: Vishnu Vijay
    - Created: 6/9/22
    - History:
        - 6/16: Adding functionality for chapter 4 proj

"""

import numpy as np
import helper as help

#####
## Initial Conditions
#####
# Initial conditions for MAV
north0 = 0.  # initial north position
east0 = 0.  # initial east position
down0 = 0.  # initial down position
u0 = 25.  # initial velocity along body x-axis
v0 = 0.  # initial velocity along body y-axis
w0 = 0.  # initial velocity along body z-axis
phi0 = 0.  # initial roll angle
theta0 = 0.  # initial pitch angle
psi0 = 0.  # initial yaw angle
p0 = 0  # initial roll rate
q0 = 0  # initial pitch rate
r0 = 0  # initial yaw rate
Va0 = np.sqrt(u0**2 + v0**2 + w0**2)
# Quaternion State
e_quat = help.EulerToQuaternion(phi0, theta0, psi0)
e0 = e_quat.item(0)
e1 = e_quat.item(1)
e2 = e_quat.item(2)
e3 = e_quat.item(3)


#####
## Physical Parameters
#####
m = 11 # kg
Jx = 0.8244 #kg m^2
Jy = 1.135 #kg m^2
Jz = 1.759 #kg m^2
Jxz = 0.1204 #kg m^2
gravity = 9.81 #m/s^2
S_wing = 0.55
b = 2.8956
c = 0.18994
S_prop = 0.2027
rho = 1.2682
e = 0.9
AR = (b**2) / S_wing


#####
## Longitudinal Coefficients
#####
C_L_0 = 0.23
C_D_0 = 0.043
C_m_0 = 0.0135
C_L_alpha = 5.61
C_D_alpha = 0.03
C_m_alpha = -2.74
C_L_q = 7.95
C_D_q = 0.0
C_m_q = -38.21
C_L_delta_e = 0.13
C_D_delta_e = 0.0135
C_m_delta_e = -0.99
M = 50.0
alpha0 = 0.47
epsilon = 0.16
C_D_p = 0.0


#####
## Lateral Coefficients
#####
C_Y_0 = 0.0
C_l_0 = 0.0
C_n_0 = 0.0
C_Y_beta = -0.98
C_l_beta = -0.13
C_n_beta = 0.073
C_Y_p = 0.0
C_l_p = -0.51
C_n_p = 0.069
C_Y_r = 0.0
C_l_r = 0.25
C_n_r = -0.095
C_Y_delta_a = 0.075
C_l_delta_a = 0.17
C_n_delta_a = -0.011
C_Y_delta_r = 0.19
C_l_delta_r = 0.0024
C_n_delta_r = -0.069


#####
## Prop/motor parameters
#####
# Prop parameters
D_prop = 20*(0.0254)     # prop diameter in m
# Motor parameters
KVstar = 145.                   # from datasheet RPM/V
KV = 60. / (2 * np.pi * KVstar)  # V-s/rad
KQ = KV  # KQ in N-m/A
R_motor = 0.042              # ohms
i0 = 1.5                     # no-load (zero-torque) current (A)
# Inputs
ncells = 12.
V_max = 3.7 * ncells  # max voltage for specified number of battery cells
# Coeffiecients from prop_data fit
C_Q2 = -0.01664
C_Q1 = 0.004970
C_Q0 = 0.005230
C_T2 = -0.1079
C_T1 = -0.06044
C_T0 = 0.09357


#####
## Calculation Variables
#####
# gamma parameters pulled from page 36 (dynamics)
gamma = Jx * Jz - (Jxz**2)
gamma1 = (Jxz * (Jx - Jy + Jz)) / gamma
gamma2 = (Jz * (Jz - Jy) + (Jxz**2)) / gamma
gamma3 = Jz / gamma
gamma4 = Jxz / gamma
gamma5 = (Jz - Jx) / Jy
gamma6 = Jxz / Jy
gamma7 = ((Jx - Jy) * Jx + (Jxz**2)) / gamma
gamma8 = Jx / gamma