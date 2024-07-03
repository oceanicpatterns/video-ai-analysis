import numpy as np

# Constants
BODY_ALIGNMENT_THRESHOLD = 0.1
LEG_MOVEMENT_THRESHOLD = 0.1
ARM_MOVEMENT_THRESHOLD = 0.3
SHOULDER_HIP_DISTANCE_THRESHOLD = 0.2
SHOULDER_WIDTH_THRESHOLD = 1.2

def calculate_angle(a, b, c):
    ba = np.array([a.x - b.x, a.y - b.y])
    bc = np.array([c.x - b.x, c.y - b.y])
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)
