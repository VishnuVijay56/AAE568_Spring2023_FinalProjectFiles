"""
mav_dynamics.py: class file for dynamics of mav
    - Author: Vishnu Vijay
    - Created: 6/9/22
    - History:
        - 6/16: Adding functionality for chapter 4 proj
        - 6/27: Adding functionality for chapter 5 proj
        - 7/14: Adding functionality for chapter 6 proj

"""

from copy import copy
import numpy as np
from helper import QuaternionToEuler, QuaternionToRotationMatrix

import mav_body_parameter as MAV_para
from mav_state import MAV_State
from delta_state import Delta_State

class MAV_Dynamics:
    ###
    # Constructor!
    # Initializes the MAVDynamics object, sets up internal and external state variables
    # internal state (internal_state) used for integration and EOMs
    # external state (mav_state) to be accessed by other files
    # Inputs:
    #   - time_step: time step to be used by simulation
    ###
    def __init__(self, time_step):
        self.time_step = time_step
        
        # Initialize internal state
        # pn, pe, pd, u, v, w, e0, e1, e2, e3, p, q, r
        self.internal_state = np.empty((13, 1))
        self.internal_state[0] = MAV_para.north0
        self.internal_state[1] = MAV_para.east0
        self.internal_state[2] = MAV_para.down0
        self.internal_state[3] = MAV_para.u0
        self.internal_state[4] = MAV_para.v0
        self.internal_state[5] = MAV_para.w0
        self.internal_state[6] = MAV_para.e0
        self.internal_state[7] = MAV_para.e1
        self.internal_state[8] = MAV_para.e2
        self.internal_state[9] = MAV_para.e3
        self.internal_state[10] = MAV_para.p0
        self.internal_state[11] = MAV_para.q0
        self.internal_state[12] = MAV_para.r0

        self.wind_i = np.zeros((3,1)) # inertial

        self.alpha = 0 # angle of attack
        self.beta = 0 # sideslip angle
        self.V_a = np.sqrt(self.internal_state.item(3)**2 + self.internal_state.item(4)**2 + self.internal_state.item(5)**2)
        self.update_velocity_data()

        self.mav_state = MAV_State()
        self.delta = Delta_State()
        self.update_external_state()
        
    
    ###
    # Iterates a singular time step, according to time_step, and calculates the new
    # state values of the MAV
    # Updates the internal and external MAV states
    # Inputs:
    #   - applied_forces_moments: array containing all the forces and moments applied to the MAV
    # Outputs: 
    # - N/A
    ###
    def iterate(self, delta, wind):
        # func call
        applied_forces_moments = self.net_force_moment(delta)
        
        # Integrate ODE using RK4 algorithm
        x1 = self.derivatives(self.internal_state, applied_forces_moments)
        x2 = self.derivatives(self.internal_state + self.time_step/2.*x1, applied_forces_moments)
        x3 = self.derivatives(self.internal_state + self.time_step/2.*x2, applied_forces_moments)
        x4 = self.derivatives(self.internal_state + self.time_step*x3, applied_forces_moments)
        self.internal_state += self.time_step/6 *(x1 + 2*x2 + 2*x3 + x4)

        # Normalize quaternion
        e0 = self.internal_state.item(6)
        e1 = self.internal_state.item(7)
        e2 = self.internal_state.item(8)
        e3 = self.internal_state.item(9)
        quaternion_mag = np.sqrt(e0**2 + e1**2 + e2**2 + e3**2)
        self.internal_state[6] = e0 / quaternion_mag
        self.internal_state[7] = e1 / quaternion_mag
        self.internal_state[8] = e2 / quaternion_mag
        self.internal_state[9] = e3 / quaternion_mag

        # update
        self.update_velocity_data(wind)

        # Update external state
        self.update_external_state()


    ###
    # Iterates a singular time step, according to time_step, and calculates the new
    # state values of the MAV
    # Inputs:
    #   - curr_state: internal state of the MAV
    #   - applied_forces_moments: array containing all the forces and moments applied to the MAV
    # Outputs:
    #   - array of derivative of all (internal) state variables
    ###
    def derivatives(self, int_state, applied_forces_moments):
        ## Extract the values of int_state and applied_forces_moments
        # int_state
        pn = int_state.item(0)
        pe = int_state.item(1)
        pd = int_state.item(2)
        u = int_state.item(3)
        v = int_state.item(4)
        w = int_state.item(5)
        e0 = int_state.item(6)
        e1 = int_state.item(7)
        e2 = int_state.item(8)
        e3 = int_state.item(9)
        p = int_state.item(10)
        q = int_state.item(11)
        r = int_state.item(12)
        # applied_forces_moments
        fx = applied_forces_moments.item(0)
        fy = applied_forces_moments.item(1)
        fz = applied_forces_moments.item(2)
        l = applied_forces_moments.item(3)
        m = applied_forces_moments.item(4)
        n = applied_forces_moments.item(5)


        ## Find derivatives
        # position kinematics
        pos_dot = QuaternionToRotationMatrix(np.array([e0, e1, e2, e3])) @ np.array([u, v, w])
        north_dot = pos_dot.item(0)
        east_dot = pos_dot.item(1)
        down_dot = pos_dot.item(2)

        # position dynamics
        u_dot = (r*v - q*w) + (fx / MAV_para.m)
        v_dot = (p*w - r*u) + (fy / MAV_para.m)
        w_dot = (q*u - p*v) + (fz / MAV_para.m)

        # rotational kinematics
        e0_dot = 0.5*(0*e0 - p*e1 - q*e2 - r*e3)
        e1_dot = 0.5*(p*e0 + 0*e1 + r*e2 - q*e3)
        e2_dot = 0.5*(q*e0 - r*e1 + 0*e2 + p*e3) 
        e3_dot = 0.5*(r*e0 + q*e1 - p*e2 + 0*e3)

        # rotatonal dynamics
        p_dot = MAV_para.gamma1*p*q - MAV_para.gamma2*q*r + MAV_para.gamma3*l + MAV_para.gamma4*n
        q_dot = MAV_para.gamma5*p*r - MAV_para.gamma6*(p**2 - r**2) + m / MAV_para.Jy
        r_dot = MAV_para.gamma7*p*q - MAV_para.gamma1*q*r + MAV_para.gamma4*l + MAV_para.gamma8*n

        # collect the derivative of the states
        x_dot = np.array([[north_dot, east_dot, down_dot, u_dot, v_dot, w_dot,
                           e0_dot, e1_dot, e2_dot, e3_dot, p_dot, q_dot, r_dot]]).T
        return x_dot


    ###
    # Updates the external MAV state based on the new internal state of the MAV
    # Inputs:
    #   - N/A
    # Outputs:
    #   - N/A
    ###
    def update_external_state(self):
        #print("  ----->>>   UPDATING EXTERNAL STATE")
        # quaternion to euler angular positions
        phi, theta, psi = QuaternionToEuler(self.internal_state[6:10])
        pdot = QuaternionToRotationMatrix(self.internal_state[6:10].T) @ self.internal_state[3:6]
        # inertial positions
        self.mav_state.north = self.internal_state.item(0)
        self.mav_state.east = self.internal_state.item(1)
        self.mav_state.altitude = -self.internal_state.item(2)
        # euler angular positions
        self.mav_state.phi = phi
        self.mav_state.theta = theta
        self.mav_state.psi = psi
        # rate of change of angular positions
        self.mav_state.p = self.internal_state.item(10)
        self.mav_state.q = self.internal_state.item(11)
        self.mav_state.r = self.internal_state.item(12)
        # flight parameters
        self.mav_state.Va = self.V_a
        self.mav_state.alpha = self.alpha
        self.mav_state.beta = self.beta
        self.mav_state.Vg = np.linalg.norm(pdot)
        self.mav_state.gamma = np.arcsin(pdot.item(2) / self.mav_state.Vg)
        self.mav_state.chi = np.arctan2(pdot.item(1), pdot.item(0))
        # wind
        self.mav_state.wn = self.wind_i.item(0)
        self.mav_state.we = self.wind_i.item(1)


    ###
    # Calculates the net force and moment vectors on the MAV
    # Inputs:
    #   - deflections: deflection of aileron, of elevator, and of rudder, and throttle level
    #
    ###
    def net_force_moment(self, deflections):
        #print("  ----->>>   UPDATING FORCE/MOMENTS")
        # Pull values from state vars
        p = self.internal_state.item(10)
        q = self.internal_state.item(11)
        r = self.internal_state.item(12)
        #print("EULER: phi (" + str(phi) + "); theta (" + str(theta) + "); psi (" + str(psi) + ")")
        #print("EULER Rates: p (" + str(p) + "); q (" + str(q) + "); r (" + str(r) + ")")
        
        # Pull values from deflections
        d_a = copy(deflections.aileron_deflection)
        d_e = copy(deflections.elevator_deflection)
        d_r = copy(deflections.rudder_deflection)
        d_t = copy(deflections.throttle_level)

        # Store delta values in global var
        self.delta.aileron_deflection = d_a
        self.delta.elevator_deflection = d_e
        self.delta.rudder_deflection = d_r
        self.delta.throttle_level = d_t
        #self.delta = deflections

        # Force of gravity
        f_g_inertial = np.array([0, 0, MAV_para.m*MAV_para.gravity])
        f_g_body = QuaternionToRotationMatrix(self.internal_state[6:10]).T @ f_g_inertial
        
        # Coefficients of Lift and Drag
        sigma = (1 + np.exp(-MAV_para.M*(self.alpha - MAV_para.alpha0)) + np.exp(MAV_para.M*(self.alpha + MAV_para.alpha0)))
        sigma = sigma / ((1 + np.exp(-MAV_para.M*(self.alpha - MAV_para.alpha0)))*(1 + np.exp(MAV_para.M*(self.alpha + MAV_para.alpha0))))
        
        
        CL = (1 - sigma)*(MAV_para.C_L_0 + MAV_para.C_L_alpha*self.alpha) 
        CL = CL + sigma*(2*(np.sin(self.alpha)**2)*(np.cos(self.alpha)))*np.sign(self.alpha)
        CL = CL + MAV_para.C_L_q*MAV_para.c*q/(2*self.V_a) + MAV_para.C_L_delta_e*d_e
        
        CD = MAV_para.C_D_p + ((MAV_para.C_L_0 + MAV_para.C_L_alpha*self.alpha)**2) / (np.pi * MAV_para.e * MAV_para.AR)
        CD = CD + MAV_para.C_D_q*MAV_para.c*q/(2*self.V_a) + MAV_para.C_D_delta_e*d_e
        
        # Elements - Force
        Tp, Qp = self.motor_thrust_torque(self.V_a, d_t)
        
        Cx = -CD*np.cos(self.alpha) + CL*np.sin(self.alpha)
        
        Cy = MAV_para.C_Y_0 + MAV_para.C_Y_beta*self.beta + MAV_para.C_Y_p*p*MAV_para.b/(2*self.V_a)
        Cy = Cy + MAV_para.C_Y_r*r*MAV_para.b/(2*self.V_a) + MAV_para.C_Y_delta_a*d_a + MAV_para.C_Y_delta_r*d_r
        
        Cz = -CD*np.sin(self.alpha) - CL*np.cos(self.alpha)
        
        # Force components
        fx = f_g_body.item(0) + Tp + (0.5*MAV_para.rho*MAV_para.S_wing*self.V_a**2)*(Cx)
        
        fy = f_g_body.item(1) + (0.5*MAV_para.rho*MAV_para.S_wing*self.V_a**2)*(Cy)
        
        fz = f_g_body.item(2) + (0.5*MAV_para.rho*MAV_para.S_wing*self.V_a**2)*(Cz)

        # Elements - Moment
        Cl = MAV_para.C_l_0 + MAV_para.C_l_beta*self.beta + MAV_para.C_l_p*MAV_para.b*p/(2*self.V_a)
        Cl = Cl + MAV_para.C_l_r*MAV_para.b*r/(2*self.V_a) + MAV_para.C_l_delta_a*d_a + MAV_para.C_l_delta_r*d_r
        
        Cm = MAV_para.C_m_0 + MAV_para.C_m_alpha*self.alpha + MAV_para.C_m_q*MAV_para.c*q/(2*self.V_a) + MAV_para.C_m_delta_e*d_e
        
        Cn = MAV_para.C_n_0 + MAV_para.C_n_beta*self.beta + MAV_para.C_n_p*MAV_para.b*p/(2*self.V_a)
        Cn = Cn + MAV_para.C_n_r*MAV_para.b*r/(2*self.V_a) + MAV_para.C_n_delta_a*d_a + MAV_para.C_n_delta_r*d_r

        # Moment components
        l = (0.5*MAV_para.rho*MAV_para.S_wing*self.V_a**2)*MAV_para.b*Cl - Qp
        
        m = (0.5*MAV_para.rho*MAV_para.S_wing*self.V_a**2)*MAV_para.c*Cm
        
        n = (0.5*MAV_para.rho*MAV_para.S_wing*self.V_a**2)*MAV_para.b*Cn 

        # Return
        f_m = np.array([fx, fy, fz, l, m, n])
        #print("FORCES: ", f_m[0:3].T)
        return f_m


    ###
    # Uses the MAV's parameters, air velocity, and throttle input to calculate the thrust and torque
    # produced by the motor.
    # Inputs:
    #   - air_vel: the velocity of air wrt the MAV
    #   - throttle: throttle input (0 to 1)
    # Outputs:
    #   - Thrust produced by motor
    #   - Torque produced by motor
    ###
    def motor_thrust_torque(self, air_vel, throttle):
        V_in = MAV_para.V_max * throttle

        a = MAV_para.rho * (MAV_para.D_prop**5) * MAV_para.C_Q0 / (4 * np.pi**2)
        b = MAV_para.rho * (MAV_para.D_prop**4) * MAV_para.C_Q1 * air_vel / (2 * np.pi) + MAV_para.KQ * MAV_para.KV / MAV_para.R_motor
        c = MAV_para.rho * (MAV_para.D_prop**3) * MAV_para.C_Q2 * (air_vel**2) - MAV_para.KQ * V_in / MAV_para.R_motor + MAV_para.KQ * MAV_para.i0

        if ((b**2 - 4*a*c) < 0.):
            #print(" --> INVALID DISCRIMINANT")
            omega_op = 0
            J_op = 0
        else:
            omega_op = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
            #if (omega_op < 0.):
                #print(" --> OMEGA_OP is NEGATIVE")
            J_op = 2 * np.pi * air_vel / (omega_op * MAV_para.D_prop)
            #if (J_op < 0.):
                #print(" --> J_OP is NEGATIVE")
        # J_op for CT = 0 : 0.692363
        

        C_T = (MAV_para.C_T2*J_op**2 + MAV_para.C_T1*J_op + MAV_para.C_T0)
        C_Q = (MAV_para.C_Q2*J_op**2 + MAV_para.C_Q1*J_op + MAV_para.C_Q0)
        # print("C_T: " + str(C_T) + "; C_Q: " + str(C_Q) + "; J_op: " + str(J_op) + "; omega_op: " + str(omega_op) + "; Va: " + str(air_vel))
        
        n = omega_op / (2*np.pi)
        T_p = MAV_para.rho * (MAV_para.D_prop**4) * (n**2) * C_T
        Q_p = MAV_para.rho * (MAV_para.D_prop**5) * (n**2) * C_Q
        #print("Thrust: " + str(T_p) + "; Torque: " + str(Q_p))
        return T_p, Q_p


    ###
    #
    #
    #
    ###
    def update_velocity_data(self, wind = np.zeros((6,1))):
        steady = wind[0:3] # in inertial frame
        gusts = wind[3:6] # already in body frame

        R = QuaternionToRotationMatrix(self.internal_state[6:10]).T[0]

        tot_wind_b = (R @ steady) + gusts
        self.wind_i = (QuaternionToRotationMatrix(self.internal_state[6:10].T) @ gusts) + steady
        

        # air velocity components wrt MAV body
        u_r = self.internal_state[3] - tot_wind_b[0] # forward
        v_r = self.internal_state[4] - tot_wind_b[1] # right
        w_r = self.internal_state[5] - tot_wind_b[2] # down

        # Airspeed
        self.V_a = np.sqrt(u_r**2 + v_r**2 + w_r**2)

        # Angle of attack computation
        if (u_r != 0):
            self.alpha = np.arctan(w_r / u_r)
        else:
            if (w_r > 0):
                self.alpha = -np.pi / 2
            elif (w_r < 0):
                self.alpha = np.pi / 2
            else:
                self.alpha = 0
        
        # Sideslip angle computation
        if (self.V_a != 0):
            self.beta = np.arcsin(v_r / (self.V_a))
        else:
            self.beta = 0