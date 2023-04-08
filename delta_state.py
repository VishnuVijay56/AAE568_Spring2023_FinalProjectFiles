"""
delta_state.py: class file for state of MAV deltas (aileron, rudder, elevator, throttle)
    - Author: Vishnu Vijay
    - Created: 6/18/22
    - History:
        - 
"""

class Delta_State:
    def __init__(self, d_e = 0., d_a = 0., d_r = 0., d_t = 0.5):
        # Aileron
        if (type(d_a) is not float):
            d_a = d_a.item(0)
        self.aileron_deflection = d_a
        
        # Elevator
        if (type(d_e) is not float):
            d_e = d_e.item(0)
        self.elevator_deflection = d_e

        # Rudder
        if (type(d_r) is not float):
            d_r = d_r.item(0)
        self.rudder_deflection = d_r

        # Throttle
        if (type(d_t) is not float):
            d_t = d_t.item(0)
        self.throttle_level = d_t

    
    def print(self):
        rounding_digits = 4
        print('DELTA: elevator =', round(self.elevator_deflection, rounding_digits),
              '; aileron =', round(self.aileron_deflection, rounding_digits),
              '; rudder =', round(self.rudder_deflection, rounding_digits),
              '; throttle =', round(self.throttle_level, rounding_digits))