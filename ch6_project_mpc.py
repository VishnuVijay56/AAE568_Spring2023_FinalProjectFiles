"""
ch6_project.py: implementing project
    - Author: Vishnu Vijay
    - Created: 7/7/22
"""

import time
import numpy as np
import matplotlib.pyplot as plt

from mav import MAV
from mav_dynamics import MAV_Dynamics
from trim import compute_trim
from compute_models import compute_model
from wind_simulation import WindSimulation
from signals import Signals

from mav_state import MAV_State
from delta_state import Delta_State
from autopilot_cmds import AutopilotCmds


# Create instance of MAV_Dynamics
Ts = 0.01
mpc_horizon = 5
mav_dynamics = MAV_Dynamics(time_step=Ts) # time step in seconds
mav_state = mav_dynamics.mav_state
# Create instance of MAV object using MAV_State object
fullscreen = False
this_mav = MAV(mav_state, fullscreen)
# Create instance of wind simulation
wind_sim = WindSimulation(Ts)
# Autopilot message
commands = AutopilotCmds()
Va_command = Signals(dc_offset=25.0,
                     amplitude=3.0,
                     start_time=2.0,
                     frequency=0.01)
altitude_command = Signals(dc_offset=0.0,
                           amplitude=15.0,
                           start_time=0.0,
                           frequency=0.02)
course_command = Signals(dc_offset=np.radians(0),
                         amplitude=np.radians(45),
                         start_time=5.0,
                         frequency=0.015)


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
from autopilot_MPC import Autopilot
autopilot = Autopilot(Ts, mpc_horizon, mav_state)

# Run Simulation
curr_time = 0
end_time = 20 # seconds
view_sim = True
sim_real_time = True
display_graphs = True

# # Print Control Parameters
# import control_parameters
# control_parameters.print_coefficients()

extra_elem = 0

if (display_graphs):
    time_arr = np.zeros(int(end_time / Ts) + extra_elem)
    
    north_history = np.zeros(int(end_time / Ts) + extra_elem)
    east_history = np.zeros(int(end_time / Ts) + extra_elem)

    alt_history = np.zeros(int(end_time / Ts) + extra_elem)
    alt_cmd_history = np.zeros(int(end_time / Ts) + extra_elem)
    
    airspeed_history = np.zeros(int(end_time / Ts) + extra_elem)
    airspeed_cmd_history = np.zeros(int(end_time / Ts) + extra_elem)
    
    phi_history = np.zeros(int(end_time / Ts) + extra_elem)
    theta_history = np.zeros(int(end_time / Ts) + extra_elem)
    psi_history = np.zeros(int(end_time / Ts) + extra_elem)

    chi_history = np.zeros(int(end_time / Ts) + extra_elem)
    chi_cmd_history = np.zeros(int(end_time / Ts) + extra_elem)
    
    d_e_history = np.zeros(int(end_time / Ts) + extra_elem)
    d_a_history = np.zeros(int(end_time / Ts) + extra_elem)
    d_r_history = np.zeros(int(end_time / Ts) + extra_elem)
    d_t_history = np.zeros(int(end_time / Ts) + extra_elem)
    
    ind = 0


while (curr_time <= end_time) and (view_sim):
    step_start = time.time()
    # print("\nTime: " + str(round(curr_time, 2)) + " ", end=" -> \n")

    # autopilot commands
    commands.airspeed_command = Va_command.square(curr_time)
    commands.course_command = course_command.square(curr_time)
    commands.altitude_command = altitude_command.square(curr_time)
    
    # autopilot
    estimated_state = mav_dynamics.mav_state  # this is the actual mav state
    delta, commanded_state = autopilot.update(commands, estimated_state)
    
    # wind sim
    wind_steady_gust = wind_sim.update() # np.zeros((6,1)) #

    # Update MAV dynamic state
    mav_dynamics.iterate(delta, wind_steady_gust)

    # Update MAV mesh for viewing
    this_mav.set_mav_state(mav_dynamics.mav_state)
    this_mav.update_mav_state()
    this_mav.update_render()
    
    # DEBUGGING - Print Vehicle's state
    if (display_graphs):
        time_arr[ind] = curr_time
        
        north_history[ind] = estimated_state.north
        east_history[ind] = estimated_state.east

        alt_history[ind] = estimated_state.altitude
        alt_cmd_history[ind] = commands.altitude_command

        airspeed_history[ind] = estimated_state.Va
        airspeed_cmd_history[ind] = commands.airspeed_command

        chi_history[ind] = estimated_state.chi * 180 / np.pi
        chi_cmd_history[ind] = commands.course_command * 180 / np.pi
        
        phi_history[ind] = estimated_state.phi * 180 / np.pi
        theta_history[ind] = estimated_state.theta * 180 / np.pi
        psi_history[ind] = estimated_state.psi * 180 / np.pi

        d_e_history[ind] = delta.elevator_deflection * 180 / np.pi
        d_a_history[ind] = delta.aileron_deflection * 180 / np.pi
        d_r_history[ind] = delta.rudder_deflection * 180 / np.pi
        d_t_history[ind] = delta.throttle_level


    #mav_dynamics.mav_state.print()
    #commands.print()
    #delta.print()
    #print("Wind: ", mav_dynamics.wind_i.T)
    #print("COMMANDED STATE:", end=" -> ")
    #commanded_state.print()

    # # Wait for 5 seconds before continuing
    # if (ind == 0):
    #     time.sleep(5.)

    # Update time
    step_end = time.time()
    if ((sim_real_time) and ((step_end - step_start) < mav_dynamics.time_step)):
        time.sleep(step_end - step_start)
    curr_time += Ts
    ind += 1


if (display_graphs):
    # Main State tracker
    fig1, axes = plt.subplots(1, 3)
    
    axes[0].plot(time_arr, alt_history)
    axes[0].plot(time_arr, alt_cmd_history)
    axes[0].legend(["True", "Command"])
    axes[0].set_title("ALTITUDE")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Altitude (meters)")

    axes[1].plot(time_arr, airspeed_history)
    axes[1].plot(time_arr, airspeed_cmd_history)
    axes[1].legend(["True", "Command"])
    axes[1].set_title("Va")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Airspeed (meters/second)")

    axes[2].plot(time_arr, chi_history)
    axes[2].plot(time_arr, chi_cmd_history)
    axes[2].legend(["True", "Command"])
    axes[2].set_title("CHI")
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