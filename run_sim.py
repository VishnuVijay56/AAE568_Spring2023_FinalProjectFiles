"""
run_sim.py: call the simulation framework, used for Monte Carlo sims
    - Author: Vishnu Vijay
    - Created: 4/23/23
"""

# Python Imports
import time
import numpy as np
import matplotlib.pyplot as plt

# User-Defined Imports : non-message
from mav import MAV
from mav_dynamics import MAV_Dynamics
from trim import compute_trim
from compute_models import compute_model
from wind_simulation import WindSimulation
from signals import Signals
import model_coef as mc

# User-Defined Imports : message
from mav_state import MAV_State
from delta_state import Delta_State
from autopilot_cmds import AutopilotCmds
from sim_cmds import SimCmds


#####
# 
#
#
#####

def run_two_plane_sim(t_span, sim_options : SimCmds):
    # Create instance of MAV_Dynamics
    Ts = 0.01
    chaser_dynamics = MAV_Dynamics(time_step=Ts) # time step in seconds
    chaser_state = chaser_dynamics.mav_state
    
    # Create instance of MAV object using MAV_State object
    fullscreen = False
    this_mav = MAV(chaser_state, fullscreen, sim_options.view_sim)
    
    # Create instance of wind simulation
    steady_state_wind = np.array([[0., 0., 0.]]).T
    wind_sim = WindSimulation(Ts, steady_state_wind, sim_options.wind_gust)
    
    # Chaser Autopilot message
    chaser_commands = AutopilotCmds()
    Va_command_chaser = Signals(dc_offset=25.0,
                                amplitude=3.0,
                                start_time=2.0,
                                frequency=0.01)
    altitude_command_chaser = Signals(dc_offset=0.0,
                                    amplitude=15.0,
                                    start_time=0.0,
                                    frequency=0.002)
    course_command_chaser = Signals(dc_offset=np.radians(0),
                                    amplitude=np.radians(45),
                                    start_time=5.0,
                                    frequency=0.0015)
    

    # # Find trim state
    # Va = 25
    # gamma_deg = 0
    # gamma_rad = gamma_deg * np.pi/180
    # trim_state, trim_input = compute_trim(mav_dynamics, Va, gamma_rad)
    # mav_dynamics.internal_state = trim_state
    # delta_trim = trim_input
    # #delta_trim.print()

    # # Compute state space model linearized about trim
    # compute_model(mav_dynamics, trim_state, trim_input)

    # Create instance of autopilot
    from autopilot_LQR import Autopilot
    from autopilot_LQR_throttle_fault import Autopilot_TF
    from autopilot_LQR_elevator_fault import Autopilot_EF
    chaser_autopilot = Autopilot(Ts)
    chaser_autopilot_tf = Autopilot_TF(Ts)
    chaser_autopilot_ef = Autopilot_EF(Ts)

    # Create instance of Kalman Filter
    from kalman_filter import KalmanFilter
    chaser_kf = KalmanFilter(chaser_state)
    estimated_chaser = chaser_state
    chaser_delta = chaser_dynamics.delta

    # Run Simulation
    curr_time = t_span[0]
    end_time = t_span[1] # seconds

    extra_elem = 0
    ind = 0

    if (sim_options.display_graphs):
        time_arr = np.zeros(int(end_time / Ts) + extra_elem)
        
        north_history = np.zeros(int(end_time / Ts) + extra_elem)
        east_history = np.zeros(int(end_time / Ts) + extra_elem)

        alt_history = np.zeros(int(end_time / Ts) + extra_elem)
        alt_history_est = np.zeros(int(end_time / Ts) + extra_elem)
        alt_history_meas = np.zeros(int(end_time / Ts) + extra_elem)
        alt_cmd_history = np.zeros(int(end_time / Ts) + extra_elem)
        
        airspeed_history = np.zeros(int(end_time / Ts) + extra_elem)
        airspeed_history_est = np.zeros(int(end_time / Ts) + extra_elem)
        airspeed_history_meas = np.zeros(int(end_time / Ts) + extra_elem)
        airspeed_cmd_history = np.zeros(int(end_time / Ts) + extra_elem)
        
        phi_history = np.zeros(int(end_time / Ts) + extra_elem)
        phi_history_est = np.zeros(int(end_time / Ts) + extra_elem)
        phi_history_meas = np.zeros(int(end_time / Ts) + extra_elem)
        theta_history = np.zeros(int(end_time / Ts) + extra_elem)
        theta_history_est = np.zeros(int(end_time / Ts) + extra_elem)
        theta_history_meas = np.zeros(int(end_time / Ts) + extra_elem)
        psi_history = np.zeros(int(end_time / Ts) + extra_elem)
        psi_history_est = np.zeros(int(end_time / Ts) + extra_elem)
        psi_history_meas = np.zeros(int(end_time / Ts) + extra_elem)

        chi_history = np.zeros(int(end_time / Ts) + extra_elem)
        chi_history_est = np.zeros(int(end_time / Ts) + extra_elem)
        chi_history_meas = np.zeros(int(end_time / Ts) + extra_elem)
        chi_cmd_history = np.zeros(int(end_time / Ts) + extra_elem)

        d_e_history = np.zeros(int(end_time / Ts) + extra_elem)
        d_a_history = np.zeros(int(end_time / Ts) + extra_elem)
        d_r_history = np.zeros(int(end_time / Ts) + extra_elem)
        d_t_history = np.zeros(int(end_time / Ts) + extra_elem)

    
    # Faults
    case = 0
    if (case == 0):
        fault_time = end_time
        switching_delay = end_time
    elif (case == 1): # throttle
        fault_time = 6
        switching_delay = 3
    elif (case == 2): # elevator
        fault_time = 1
        switching_delay = 3
    
    fault_coord = None
    final_coord = None

    sim_start = time.time()

    while (curr_time <= end_time):
        step_start = time.time()
        #print("\nTime: " + str(round(curr_time, 2)) + " ", end=" -> \n")

        # Chaser: autopilot commands
        chaser_commands.airspeed_command = Va_command_chaser.square(curr_time)
        chaser_commands.course_command = course_command_chaser.square(curr_time)
        chaser_commands.altitude_command = altitude_command_chaser.square(curr_time)
        
        # Kalman Filter
        if (not sim_options.use_kf):
            estimated_chaser = chaser_dynamics.mav_state #this is the actual mav state
            measured_state = estimated_chaser
        else:
            estimated_chaser, measured_state = chaser_kf.update(chaser_state, chaser_delta)
        

        # Autopilot Switching
        if (case == 1) and (curr_time > fault_time + switching_delay):
            chaser_delta, commanded_state = chaser_autopilot_tf.update(chaser_commands, estimated_chaser)
            
            if (fault_coord == None):
                fault_coord = (chaser_state.north, chaser_state.east, chaser_state.altitude)

        elif (case == 2) and (curr_time > fault_time + switching_delay):
            chaser_delta, commanded_state = chaser_autopilot_ef.update(chaser_commands, estimated_chaser)
            
            if (fault_coord == None):
                fault_coord = (chaser_state.north, chaser_state.east, chaser_state.altitude)

        else: 
            chaser_delta, commanded_state = chaser_autopilot.update(chaser_commands, estimated_chaser)
        

        # Sim fault
        if (case == 1) and (curr_time > fault_time): # throttle
            chaser_delta.throttle_level = 0

        elif (case == 2) and (curr_time > fault_time): # elevator
            chaser_delta.elevator_deflection = saturate(chaser_delta.elevator_deflection, np.radians(-5), np.radians(5))
            

        # wind sim
        wind_steady_gust =  wind_sim.update() # np.zeros((6,1)) #


        # Update MAV dynamic state
        chaser_dynamics.iterate(chaser_delta, wind_steady_gust)
        chaser_state = chaser_dynamics.mav_state


        # Update MAV mesh for viewing
        this_mav.set_mav_state(chaser_dynamics.mav_state)
        this_mav.update_mav_state()
        if(this_mav.view_sim):
            this_mav.update_render()
        

        # DEBUGGING - Print Vehicle's state
        if (sim_options.display_graphs):
            time_arr[ind] = curr_time
            
            north_history[ind] = chaser_state.north
            east_history[ind] = chaser_state.east

            alt_history[ind] = chaser_state.altitude
            alt_history_est[ind] = estimated_chaser.altitude
            alt_history_meas[ind] = measured_state.altitude
            alt_cmd_history[ind] = chaser_commands.altitude_command

            airspeed_history[ind] = chaser_state.Va
            airspeed_history_est[ind] = estimated_chaser.Va
            airspeed_history_meas[ind] = measured_state.Va
            airspeed_cmd_history[ind] = chaser_commands.airspeed_command

            chi_history[ind] = chaser_state.chi * 180 / np.pi
            chi_history_est[ind] = estimated_chaser.chi * 180 / np.pi
            chi_history_meas[ind] = measured_state.chi * 180 / np.pi
            chi_cmd_history[ind] = chaser_commands.course_command * 180 / np.pi
            
            phi_history[ind] = chaser_state.phi * 180 / np.pi
            phi_history_est[ind] = estimated_chaser.phi * 180 / np.pi
            theta_history[ind] = chaser_state.theta * 180 / np.pi
            theta_history_est[ind] = estimated_chaser.theta * 180 / np.pi
            psi_history[ind] = chaser_state.psi * 180 / np.pi
            psi_history_est[ind] = estimated_chaser.psi * 180 / np.pi

            d_e_history[ind] = chaser_delta.elevator_deflection * 180 / np.pi
            d_a_history[ind] = chaser_delta.aileron_deflection * 180 / np.pi
            d_r_history[ind] = chaser_delta.rudder_deflection * 180 / np.pi
            d_t_history[ind] = chaser_delta.throttle_level

        # # Wait for 5 seconds before continuing
        # if (ind == 0):
        #     time.sleep(5.)

        # Update time
        step_end = time.time()
        if ((sim_options.sim_real_time) and ((step_end - step_start) < chaser_dynamics.time_step)):
            time.sleep(step_end - step_start)
        curr_time += Ts
        ind += 1

    print("Simulation Time: ", time.time() - sim_start)


    ## Glide Path Angle Calculations
    if (case != 0):
        final_coord = (chaser_state.north, chaser_state.east, chaser_state.altitude)
        alt_diff = fault_coord[2] - final_coord[2]
        ground_dist = np.sqrt(np.power(fault_coord[0] - final_coord[0], 2) + np.power(fault_coord[1] - final_coord[1], 2))
        glide_angle = np.arctan2(alt_diff, ground_dist)
        print("Fault: ", fault_coord, " --> ", final_coord)
        print("\nGLIDE ANGLE (deg): ", np.rad2deg(glide_angle))
        print()
    

    if (sim_options.display_graphs):
        # Main State tracker
        fig1, axes = plt.subplots(3, 1)
        
        axes[0].plot(time_arr, alt_history_meas, 'm+')
        axes[0].plot(time_arr, alt_history, 'c')
        axes[0].plot(time_arr, alt_cmd_history, 'k')
        axes[0].plot(time_arr, alt_history_est, 'r')
        axes[0].legend(["Measured", "True",  "Command",  "Estimate"])
        # axes[0].set_title("ALTITUDE")
        axes[0].set_xlabel("Time (seconds)")
        axes[0].set_ylabel("Altitude (meters)")

        axes[1].plot(time_arr, airspeed_history_meas, 'm+')
        axes[1].plot(time_arr, airspeed_history, 'c')
        axes[1].plot(time_arr, airspeed_cmd_history, 'k')
        axes[1].plot(time_arr, airspeed_history_est, 'r')
        axes[1].legend(["Measured", "True",  "Command",  "Estimate"])
        # axes[1].set_title("Va")
        axes[1].set_xlabel("Time (seconds)")
        axes[1].set_ylabel("Airspeed (meters/second)")

        axes[2].plot(time_arr, chi_history_meas, 'm+')
        axes[2].plot(time_arr, chi_history, 'c')
        axes[2].plot(time_arr, chi_cmd_history, 'k')
        axes[2].plot(time_arr, chi_history_est, 'r')
        axes[2].legend(["Measured", "True",  "Command",  "Estimate"])
        # axes[2].set_title("CHI")
        axes[2].set_xlabel("Time (seconds)")
        axes[2].set_ylabel("Course Heading (degrees)")

        # Inputs
        fig2, axes2 = plt.subplots(2,2)

        axes2[0,0].plot(time_arr, d_e_history)
        axes2[0,0].set_title("ELEVATOR")
        axes2[0,0].set_xlabel("Time (seconds)")
        axes2[0,0].set_ylabel("Deflection (degrees)")

        axes2[1,0].plot(time_arr, d_a_history)
        axes2[1,0].set_title("AILERON")
        axes2[1,0].set_xlabel("Time (seconds)")
        axes2[1,0].set_ylabel("Deflection (degrees)")

        axes2[0,1].plot(time_arr, d_t_history)
        axes2[0,1].set_title("THROTTLE")
        axes2[0,1].set_xlabel("Time (seconds)")
        axes2[0,1].set_ylabel("Level (percent)")

        axes2[1,1].plot(time_arr, d_r_history)
        axes2[1,1].set_title("RUDDER")
        axes2[1,1].set_xlabel("Time (seconds)")
        axes2[1,1].set_ylabel("Deflection (degrees)")


        # Plane tracking
        fig2 = plt.figure()

        ax = fig2.add_subplot(2, 2, (1,4), projection='3d')
        ax.plot3D(east_history, north_history, alt_history)
        ax.set_title("MAV POSITION TRACK")
        ax.set_xlabel("East Position (meters)")
        ax.set_ylabel("North Position (meters)")
        ax.set_zlabel("Altitude (meters)")
        set_axes_equal(ax)

        # Show plots
        plt.show()


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

#####
#
#
#
#####

# def compute_score(true_leader_state, true_chaser_state):


def saturate(input, low_limit, up_limit):
    if input <= low_limit:
        output = low_limit
    elif input >= up_limit:
        output = up_limit
    else:
        output = input
    return output