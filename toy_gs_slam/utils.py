import torch
import numpy as np
import matplotlib

DEVICE = "cpu"

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device=DEVICE)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device=DEVICE)
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def getProjectionMatrix(znear, zfar, fovX, fovY):
    import math
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def focal2fov(focal, pixels):
    import math
    return 2*math.atan(pixels/(2*focal))

def homogeneous(points):
    """
    homogeneous points
    :param points: [..., 3]
    """
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)

def homogeneous_vec(vec):
    """
    homogeneous points
    :param points: [..., 3]
    """
    return torch.cat([vec, torch.zeros_like(vec[..., :1])], dim=-1)

def toy_gaussian_model(num_points):
    length = 0.5
    x = np.linspace(-1, 1, num_points) * length
    y = np.linspace(-1, 1, num_points) * length
    x, y = np.meshgrid(x, y)
    means3D = torch.from_numpy(np.stack([x,y, 0 * np.random.rand(*x.shape)], axis=-1).reshape(-1,3)).to(DEVICE).float()
    quats = torch.zeros(1,4).repeat(len(means3D), 1).to(DEVICE)
    quats[..., 0] = 1. # tangent vectors are the canonical basis 
    scale = length /(num_points-1) 
    scales = torch.zeros(1,3).repeat(len(means3D), 1).fill_(scale).to(DEVICE) # ellipsoids without deformation i.e. circles 

    opacity = torch.ones_like(means3D[:,:1])
    colors  = matplotlib.colormaps['Accent'](np.random.randint(1,64, 64)/64)[..., :3]
    colors  = torch.from_numpy(colors).to(DEVICE)
    intrins = torch.tensor([
    [711.1111,   0.0000, 256.0000,   0.0000],
    [  0.0000, 711.1111, 256.0000,   0.0000],
    [  0.0000,   0.0000,   1.0000,   0.0000],
    [  0.0000,   0.0000,   0.0000,   1.0000]
    ]).to(DEVICE)

    intrins = intrins[:3, :3]
    projmat = torch.eye(4, dtype=torch.float)
    projmat[:3,:3] = intrins
    projmat[-1,-2] = 1.0
    projmat = projmat.T

    return {"means3D"   : means3D, 
            "scales"    : scales, 
            "squats"    : quats,
            "projmat"   : projmat,
            "colors"    : colors,
            "opacities" : opacity, 
            "intrins"   : intrins}
            