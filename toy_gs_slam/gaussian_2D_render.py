import torch
import numpy as np 

from tiny_renderer.utils import build_scaling_rotation, homogeneous, homogeneous_vec
from tiny_renderer.utils import getProjectionMatrix, focal2fov
from tiny_renderer.utils_render import alpha_blending_with_gaussians

    # def __init__(self, device="cpu"):
    #     self.device    = device
    #     self.means3D   = None
    #     self.scales    = None
    #     self.quats     = None
    #     self.opacities = None
    #     self.colors    = None
    #     self.viewmat   = None
    #     self.projmat   = None
    #     self.intrins   = None

def setup(means3D, scales, quats, viewmat, projmat, colors, opacities, device):
    rotations = build_scaling_rotation(scales, quats).permute(0,2,1)

    # 1. Viewing transform
    p_view = (means3D @ viewmat[:3,:3]) + viewmat[-1:,:3]
    uv_view = (rotations @ viewmat[:3,:3])
    M = torch.cat([homogeneous_vec(uv_view[:,:2,:]), homogeneous(p_view.unsqueeze(1))], dim=1)

    T = M @ projmat 

    # 2. Compute AABB
    temp_point = torch.tensor([[1.,1., -1.]]).to(device)
    distance = (temp_point * (T[..., 3] * T[..., 3])).sum(dim=-1, keepdims=True)
    f = (1 / distance) * temp_point
    point_image = torch.cat(
        [(f * T[..., 0] * T[...,3]).sum(dim=-1, keepdims=True),
        (f * T[..., 1] * T[...,3]).sum(dim=-1, keepdims=True),
        (f * T[..., 2] * T[...,3]).sum(dim=-1, keepdims=True)], dim=-1)

    half_extend = point_image * point_image - torch.cat(
        [(f * T[..., 0] * T[...,0]).sum(dim=-1, keepdims=True),
        (f * T[..., 1] * T[...,1]).sum(dim=-1, keepdims=True),
        (f * T[..., 2] * T[...,2]).sum(dim=-1, keepdims=True)], dim=-1)

    radii = half_extend.clamp(min=1e-4).sqrt() * 3 # three sigma
    center = point_image

    # 3. Perform Sorting
    depth = p_view[..., 2] # depth is used only for sorting
    index = depth.sort()[1]
    T = T[index]
    colors = colors[index]
    center = center[index]
    depth = depth[index]
    radii = radii[index]
    opacities = opacities[index]
    return T, colors, opacities, center, depth, radii

def surface_splatting(means3D, scales, quats, viewpoint, projmat, colors, opacities, intrins, device):

    viewmat = torch.eye(4)
    viewmat[:3, :3] = viewpoint.R
    viewmat[:3, 3]  = viewpoint.T
    # Rasterization setup
    projmat = torch.zeros(4,4).to(device)
    projmat[:3,:3] = intrins
    projmat[-1,-2] = 1.0
    projmat = projmat.T
    T, colors, opacities, center, depth, radii = setup(means3D, scales, quats, viewmat, projmat, colors, opacities, device)

    # Rasterization
    # 1. Generate pixels
    W, H = (intrins[0,-1] * 2).long(), (intrins[1,-1] * 2).long()
    W, H = W.item(), H.item()
    pix = torch.stack(torch.meshgrid(torch.arange(W),
        torch.arange(H), indexing='xy'), dim=-1).to(device)

    # 2. Compute ray splat intersection # Eq.9 and Eq.10
    x = pix.reshape(-1,1,2)[..., :1]
    y = pix.reshape(-1,1,2)[..., 1:]
    k = -T[None][..., 0] + x * T[None][..., 3] #<-------------
    l = -T[None][..., 1] + y * T[None][..., 3] #<-------------
    points = torch.cross(k, l, dim=-1)
    s = points[..., :2] / points[..., -1:]

    # 3. add low pass filter # Eq. 11
    # when a point (2D Gaussian) viewed from a far distance or from a slended angle
    # the 2D Gaussian will falls between pixels and no fragment is used to rasterize the Gaussian
    # so we should add a low pass filter to handle such aliasing.
    dist3d = (s * s).sum(dim=-1)
    filtersze = np.sqrt(2) / 2
    dist2d = (1/filtersze)**2 * (torch.cat([x,y], dim=-1) - center[None,:,:2]).norm(dim=-1)**2 #<-------
    # min of dist2 is equal to max of Gaussian exp(-0.5 * dist2)
    dist2 = torch.min(dist3d, dist2d)
    # dist2 = dist3d
    depth_acc = (homogeneous(s) * T[None,..., -1]).sum(dim=-1) #<--------

    # 4. accumulate 2D gaussians through alpha blending # Eq.12
    image, depthmap = alpha_blending_with_gaussians(dist2, colors, opacities, depth_acc, H, W) #<---------

    return {
        "render" : image, 
        "depth"  : depthmap, 
        "opacity": opacities,
        "center" : center, 
        "radii"  : radii, 
        "dist2"  : dist2
    } #<----------
