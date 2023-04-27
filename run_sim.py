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

# User-Defined Imports : message
from mav_state import MAV_State
from delta_state import Delta_State
from autopilot_cmds import AutopilotCmds
from sim_cmds import SimCmds
from kalman_filter import KalmanFilter


#####
# 
#
#
#####
def error_calculation(vector_1, vector_2):
    error_vs_time = vector_1 - vector_2
    error_total = np.sum(error_vs_time**2)
    return error_total

def run_two_plane_sim(t_span, sim_options : SimCmds):
    # General Definitions
    Ts = 0.01
    mpc_horizon = 25
    controller_type = "MPC"  # "LQR", or "MPC"

    curr_time = t_span[0]
    end_time = t_span[1]  # seconds

    # Faults
    case = 0
    if (case == 0):
        fault_time = end_time
        switching_delay = end_time
    elif(case ==1):  # Throttle
        fault_time = 6
        switching_delay = 3
    elif (case == 2):  # Elevator
        fault_time = 1
        switching_delay = 3
    fault_coord = None
    final_coord = None


    # Create instance of MAV_Dynamics
    chaser_dynamics = MAV_Dynamics(time_step=Ts) # time step in seconds

    chaser_state = chaser_dynamics.mav_state
    leader_dynamics = MAV_Dynamics(time_step=Ts) # time step in seconds

    chaser_state = chaser_dynamics.mav_state
    
    # Create instance of MAV object using MAV_State object
    fullscreen = False
    this_mav = MAV(chaser_state, fullscreen, sim_options.view_sim)
    
    # Create instance of wind simulation
    steady_state_wind = np.array([[0., 0., 0.]]).T
    wind_sim = WindSimulation(Ts, steady_state_wind, sim_options.wind_gust)
    
    # Chaser Autopilot message
    chaser_commands = AutopilotCmds()
    Va_command_chaser = Signals(dc_offset=22.0,
                                amplitude=3.0,
                                start_time=2.0,
                                frequency=0.001)
    altitude_command_chaser = Signals(dc_offset=0.0,
                                    amplitude=15.0,
                                    start_time=0.0,
                                    frequency=0.002)
    course_command_chaser = Signals(dc_offset=np.radians(0),
                                    amplitude=np.radians(45),
                                    start_time=5.0,
                                    frequency=0.0015)

    # Create instance of autopilot
    from autopilot_LQR import Autopilot

    if controller_type == "LQR":
        from autopilot_LQR import Autopilot
        from autopilot_LQR_elevator_fault import Autopilot_EF
        from autopilot_LQR_throttle_fault import Autopilot_TF

        chaser_autopilot_tf = Autopilot_TF(Ts)
        chaser_autopilot_ef = Autopilot_EF(Ts)
        chaser_autopilot = Autopilot(Ts)
    elif controller_type == "MPC":
        from autopilot_MPC import Autopilot_MPC
        from autopilot_MPC_throttle_fault import Autopilot_MPC_TF
        from autopilot_MPC_elevator_fault import Autopilot_MPC_EF

        chaser_autopilot_tf = Autopilot_MPC_TF(Ts, mpc_horizon, chaser_state)
        chaser_autopilot_ef = Autopilot_MPC_EF(Ts, mpc_horizon, chaser_state)
        chaser_autopilot = Autopilot_MPC(Ts, mpc_horizon, chaser_state)
    else:
        print("Goofy Ahh")

    leader_autopilot = Autopilot(Ts)
    kf = KalmanFilter(chaser_state)

    # Run Simulation
    curr_time = t_span[0]
    end_time = t_span[1] # seconds

    extra_elem = 0

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
        
    ind = 0

    time_start = time.time()

    chaser_delta = chaser_dynamics.delta
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
        else:
            estimated_chaser, measured_state = kf.update(chaser_state, chaser_delta)

        # Autopilot
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






        # wind sim
        wind_steady_gust = wind_sim.update()  # np.zeros((6,1))  #

        # Update MAV dynamic state
        chaser_dynamics.iterate(chaser_delta, wind_steady_gust)

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

    time_end = time.time()

    ## Glide Path Angle Calculations
    if (case != 0):
        final_coord = (chaser_state.north, chaser_state.east, chaser_state.altitude)
        alt_diff = fault_coord[2] - final_coord[2]
        ground_dist = np.sqrt(
            np.power(fault_coord[0] - final_coord[0], 2) + np.power(fault_coord[1] - final_coord[1], 2))
        glide_angle = np.arctan2(alt_diff, ground_dist)
        print("Fault: ", fault_coord, " --> ", final_coord)
        print("\nGLIDE ANGLE (deg): ", np.rad2deg(glide_angle))
        print()

    if (sim_options.display_graphs):
        altitude_error = error_calculation(alt_history, alt_cmd_history)
        airspeed_error = error_calculation(airspeed_history, airspeed_cmd_history)
        chi_error = error_calculation(chi_history, chi_cmd_history)

        print("Altitude Error: ", altitude_error)
        print("Airspeed Error: ", airspeed_error)
        print("Chi Error: ", chi_error)

    if (sim_options.display_graphs):
        # Main State tracker
        fig1, axes = plt.subplots(3, 1)
        # Old
        # axes[0].plot(time_arr, alt_history)
        # axes[0].plot(time_arr, alt_cmd_history)
        # axes[0].legend(["True", "Command"])
        # axes[0].set_title("ALTITUDE")
        # axes[0].set_xlabel("Time (seconds)")
        # axes[0].set_ylabel("Altitude (meters)")
        #
        # axes[1].plot(time_arr, airspeed_history)
        # axes[1].plot(time_arr, airspeed_cmd_history)
        # axes[1].legend(["True", "Command"])
        # axes[1].set_title("Va")
        # axes[1].set_xlabel("Time (seconds)")
        # axes[1].set_ylabel("Airspeed (meters/second)")
        #
        # axes[2].plot(time_arr, chi_history)
        # axes[2].plot(time_arr, chi_cmd_history)
        # axes[2].legend(["True", "Command"])
        # axes[2].set_title("CHI")
        # axes[2].set_xlabel("Time (seconds)")
        # axes[2].set_ylabel("Course Heading (degrees)")

        # New
        # Main State tracker
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

        # Show plots
        plt.show()
    print("End Time: ", time_end - time_start)
    print("##########################")



#####
#
#
#
#####

# def compute_score(true_leader_state, true_chaser_state):
