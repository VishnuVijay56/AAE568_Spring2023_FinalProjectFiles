"""
autopilot_cmds.py: command/message type for autopilot
    - Author: Vishnu Vijay
    - Created: 7/13/22
"""

class AutopilotCmds:
    def __init__(self):
        self.airspeed_command = 0.0  # commanded airspeed m/s
        self.course_command = 0.0  # commanded course angle in rad
        self.altitude_command = 0.0  # commanded altitude in m
        self.phi_feedforward = 0.0  # feedforward command for roll angle

    def print(self):
        rounding_digits = 4
        print('COMMANDS: Va =', round(self.airspeed_command, rounding_digits),
              '; Chi =', round(self.course_command, rounding_digits),
              '; Alt =', round(self.altitude_command, rounding_digits),
              '; Phi =', round(self.phi_feedforward, rounding_digits))