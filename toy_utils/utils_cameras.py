import numpy as np
import torch

DEVICE = 'cpu'

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    rm = np.array([[a*a + b*b - c*c - d*d, 2*(b*c - a*d),     2*(b*d + a*c)],
                   [2*(b*c + a*d),     a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
                   [2*(b*d - a*c),     2*(c*d + a*b),     a*a + d*d - b*b - c*c]])
    return torch.from_numpy(rm)

def look_at(eye, center, up):
    """
    Create a view matrix looking from eye to center with the given up vector.
    """
    eye = np.array(eye, dtype=np.float32)
    center = np.array(center, dtype=np.float32)
    up = np.array(up, dtype=np.float32)

    f = center - eye
    f /= np.linalg.norm(f)
    up /= np.linalg.norm(up)
    s = np.cross(f, up)
    u = np.cross(s, f)

    view_matrix = np.eye(4, dtype=np.float32)
    view_matrix[:3, 0] = s
    view_matrix[:3, 1] = u
    view_matrix[:3, 2] = -f
    view_matrix[:3, 3] = eye
    return torch.from_numpy(view_matrix)

def generate_circular_trajectory(center, radius, num_steps, up=[0, 1, 0]):
    """
    Generate camera poses along a circular trajectory around a center point.
    """
    cameras = []
    for t in np.linspace(0, 2 * np.pi, num_steps, endpoint=False):
        x = center[0] + radius * np.cos(t)
        y = center[1]
        z = center[2] + radius * np.sin(t)
        eye = [x, y, z]
        view_matrix = look_at(eye, center, up)
        cameras.append(view_matrix)
    return cameras

def generate_circular_trajectory_xy(center, radius, num_steps, up=[0, 0, 1]):
    """
    Generate camera poses along a circular trajectory around a center point.
    """
    cameras = []
    for t in np.linspace(0, 2 * np.pi, num_steps, endpoint=False):
        x = center[0] + radius * np.cos(t)
        y = center[1] + radius * np.sin(t)
        z = center[2] 
        eye = [x, y, z]
        view_matrix = look_at(eye, np.array([0, 0, 0]), up)
        cameras.append(view_matrix)
    return cameras
