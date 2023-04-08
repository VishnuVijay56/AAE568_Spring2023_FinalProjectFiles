"""
helper.py: collection of helper functions for chapter projects
    - Author: Vishnu Vijay
    - Created: 6/1/22
    - History:
        - 6/7: Adding functionality for chapter 3 project
        - 6/16: Adding functionality for chapter 4 project
"""

import numpy as np

###
# Returns rotation matrix based upon 3-2-1 Euler
# angle rotation sequence
# Inputs:
#   - phi: roll angle
#   - theta: pitch angle
#   - psi: heading (from north)
# Outputs:
#   - Rotation matrix from vehicle to body frame
###
def EulerRotationMatrix(phi, theta, psi):
    # Rotation matrix from vehicle frame to vehicle-1 frame
    c_psi = np.cos(psi)
    s_psi = np.sin(psi)
    R_v_v1 = np.array([[c_psi, s_psi, 0], [-s_psi, c_psi, 0], [0, 0, 1]], dtype='object')

    # Rotation matrix from vehicle-1 frame to vehicle-2 frame
    c_the = np.cos(theta)
    s_the = np.sin(theta)
    R_v1_v2 = np.array([[c_the, 0, -s_the], [0, 1, 0], [s_the, 0, c_the]], dtype='object')

    # Rotation matrix from vehicle-2 frame to body frame
    c_phi = np.cos(phi)
    s_phi = np.sin(phi)
    R_v2_b = np.array([[1, 0, 0], [0, c_phi, s_phi], [0, -s_phi, c_phi]], dtype='object')

    # Rotation matrix from vehicle frame to body frame
    R_v_b = R_v2_b @ R_v1_v2 @ R_v_v1

    return R_v_b


###
# Returns unit quaternion representation of the euler angles
# passed to the function
# Inputs:
#   - phi: roll angle
#   - theta: pitch angle
#   - psi: heading (from north)
# Outputs:
#   - Unit quaternion
###
def EulerToQuaternion(phi, theta, psi):
    # Cosine and Sine of the half angles of psi, theta, and phi
    c_phi = np.cos(phi / 2)
    c_the = np.cos(theta / 2)
    c_psi = np.cos(psi / 2)

    s_phi = np.sin(phi / 2)
    s_the = np.sin(theta / 2)
    s_psi = np.sin(psi / 2)
    

    # Quaternion elements
    e0 = c_psi * c_the * c_phi + s_psi * s_the * s_phi
    e1 = c_psi * c_the * s_phi - s_psi * s_the * c_phi
    e2 = c_psi * s_the * c_phi + s_psi * c_the * s_phi
    e3 = s_psi * c_the * c_phi - c_psi * s_the * s_phi

    quaternion_mag = np.sqrt(e0**2 + e1**2 + e2**2 + e3**2)
    e = np.array([e0, e1, e2, e3])/ quaternion_mag

    return e


###
# Returns Euler angle position representation of the quaternion
# passed to the function
# Inputs:
#   - e_quaternion: unit quaternion
# Outputs: 
#   - phi: roll angle
#   - theta: pitch angle
#   - psi: heading angle
###
def QuaternionToEuler(e_quaternion):
    # Extract quaternion elements
    e0 = e_quaternion[0]
    e1 = e_quaternion[1]
    e2 = e_quaternion[2]
    e3 = e_quaternion[3]
    quaternion_mag = np.sqrt(e0**2 + e1**2 + e2**2 + e3**2)
    e0 /= quaternion_mag
    e1 /= quaternion_mag
    e2 /= quaternion_mag
    e3 /= quaternion_mag

    # Calculate angular positions
    phi = np.arctan2(2*(e0*e1 + e2*e3), (1 - 2*(e1**2 + e2**2))) #(e0**2 + e3**2 - e1**2 - e2**2))
    theta = np.arcsin(2*(e0*e2 - e1*e3))
    psi = np.arctan2(2*(e0*e3 + e1*e2), (1 - 2*(e2**2 + e3**2))) #(e0**2 + e1**2 - e2**2 - e3**2))

    return phi, theta, psi

###
# Returns rotation matrix from body to inertial frame
# Inputs:
#   - e_quaternion: unit quaternion
# Outputs: 
#   - Rotation matrix from body to inertial
###
def QuaternionToRotationMatrix(e_quaternion):
    # Extract quaternion elements
    e0 = e_quaternion.item(0)
    e1 = e_quaternion.item(1)
    e2 = e_quaternion.item(2)
    e3 = e_quaternion.item(3)
    
    # Create an empty 3x3 matrix
    # R_b_i = np.array([[e0**2 + e1**2 - e2**2 - e3**2, 2*(e1*e2 - e0*e3), 2*(e1*e3 + e0*e2)],
    #               [2*(e1*e2 + e0*e3), e0**2 - e1**2 + e2**2 - e3**2, 2*(e2*e3 - e0*e1)],
    #               [2*(e1*e3 - e0*e2), 2*(e2*e3 + e0*e1), e0**2 - e1**2 - e2**2 + e3**2]
    #               ])
    
    R_b_i = np.array([[(1 - 2*e2**2 - 2*e3**2), 2*(e1*e2 - e0*e3), 2*(e1*e3 + e0*e2)],
                      [2*(e1*e2 + e0*e3), (1 - 2*e1**2 - 2*e3**2), 2*(e2*e3 - e0*e1)],
                      [2*(e1*e3 - e0*e2), 2*(e2*e3 + e0*e1), (1 - 2*e1**2 - 2*e2**2)]])
    
    return R_b_i