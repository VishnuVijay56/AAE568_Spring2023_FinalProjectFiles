"""
saturate_cmds.py: command/message type for saturation of system actuators
    - Author: Vishnu Vijay
    - Created: 4/26/23
"""

class SaturateCmds:
    def __init__(self, de_span, da_span, dr_span, dt_span):
        self.min_de = saturate(de_span[0], -30, 30)
        self.max_de = saturate(de_span[1], -30, 30)
        self.min_da = saturate(da_span[0], -30, 30)
        self.max_da = saturate(da_span[1], -30, 30)
        self.min_dr = saturate(dr_span[0], -30, 30)
        self.max_dr = saturate(dr_span[1], -30, 30)
        self.min_dt = saturate(dt_span[0], 0, 1)
        self.max_dt = saturate(dt_span[1], 0, 1)

def saturate(input, low_limit, up_limit):
    if input <= low_limit:
        output = low_limit
    elif input >= up_limit:
        output = up_limit
    else:
        output = input
    return output