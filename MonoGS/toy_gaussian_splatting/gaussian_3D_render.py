import torch 

from toy_utils.utils import build_scaling_rotation, homogeneous, homogeneous_vec
from toy_utils.utils_render import alpha_blending_with_gaussians
from utils.pose_utils import SE3_exp

# DEVICE = 'cpu'

def build_covariance_2d(
    mean3d, cov3d, viewmatrix, tan_fovx, tan_fovy, focal_x, focal_y
):
    import math
    t = (mean3d @ viewmatrix[:3,:3]) + viewmatrix[-1:,:3]
    tz = t[..., 2]
    tx = t[..., 0]
    ty = t[..., 1]

    # Eq.29 locally affine transform
    # perspective transform is not affine so we approximate with first-order taylor expansion
    # notice that we multiply by the intrinsic so that the variance is at the sceen space
    J = torch.zeros(mean3d.shape[0], 3, 3).to(mean3d)
    J[..., 0, 0] = 1 / tz * focal_x
    J[..., 0, 2] = -tx / (tz * tz) * focal_x
    J[..., 1, 1] = 1 / tz * focal_y
    J[..., 1, 2] = -ty / (tz * tz) * focal_y
    W = viewmatrix[:3,:3].T # transpose to correct viewmatrix
    cov2d = J @ W @ cov3d @ W.T @ J.permute(0,2,1)

    # add low pass filter here according to E.q. 32
    filter = torch.eye(2,2).to(cov2d) * 0.0
    return cov2d[:, :2, :2] + filter[None]

def build_covariance_3d(s, r):
    L = build_scaling_rotation(s, r).permute(0,2,1)
    actual_covariance = L @ L.transpose(1, 2)
    return actual_covariance

def projection_ndc(points, viewmatrix, projmatrix):
    points_o = homogeneous(points) # object space
    points_h = points_o @ viewmatrix @ projmatrix # screen space # RHS
    p_w = 1.0 / (points_h[..., -1:] + 0.000001)
    p_proj = points_h * p_w
    p_view = points_o @ viewmatrix
    in_mask = p_view[..., 2] >= 0.2
    return p_proj, p_view, in_mask

def get_radius(cov2d):
    det = cov2d[:, 0, 0] * cov2d[:,1,1] - cov2d[:, 0, 1] * cov2d[:,1,0]
    mid = 0.5 * (cov2d[:, 0,0] + cov2d[:,1,1])
    lambda1 = mid + torch.sqrt((mid**2-det).clip(min=0.1))
    lambda2 = mid - torch.sqrt((mid**2-det).clip(min=0.1))
    return 3.0 * torch.sqrt(torch.max(lambda1, lambda2)).ceil()

def volume_splatting(means3D, scales, quats, viewpoint, projmat, colors, opacities, intrins, device):
    
    #viewmat = torch.eye(4)
    #viewmat[:3, :3] = viewpoint.R
    #viewmat[-1:,:3] = viewpoint.T

    tau = torch.cat([viewpoint.cam_trans_delta, viewpoint.cam_rot_delta], axis=0)

    viewmat = torch.eye(4, device=tau.device)
    viewmat[0:3, 0:3] = viewpoint.R
    viewmat[0:3, 3] = viewpoint.T

    viewmat = SE3_exp(tau) @ viewmat

    projmat = torch.zeros(4,4).to(device)
    projmat[:3,:3] = intrins
    projmat[-1,-2] = 1.0
    projmat = projmat.T

    mean_ndc, mean_view, in_mask = projection_ndc(means3D, viewmatrix=viewmat.T, projmatrix=projmat)

    depths = mean_view[:,2]
    mean_coord_x = mean_ndc[..., 0]
    mean_coord_y = mean_ndc[..., 1]

    means2D = torch.stack([mean_coord_x, mean_coord_y], dim=-1)
    # scales = torch.cat([scales[..., :2], scales[..., -1:]*1e-2], dim=-1)
    cov3d = build_covariance_3d(scales, quats)

    W, H = (intrins[0,-1] * 2).long().item(), (intrins[1,-1] * 2).long().item()
    fx, fy = intrins[0,0], intrins[1,1]
    tan_fovx = W / (2 * fx)
    tan_fovy = H / (2 * fy)
    cov2d = build_covariance_2d(means3D, cov3d, viewmat.T, tan_fovx, tan_fovy, fx, fy)
    radii = get_radius(cov2d)

    # Rasterization
    # generate pixels
    pix = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W), indexing='xy'), dim=-1).to(device).flatten(0,-2)
    sorted_conic = cov2d.inverse() # inverse of variance
    dx = (pix[:,None,:] - means2D[None,:]) # B P 2
    dist2 = dx[:, :, 0]**2 * sorted_conic[:, 0, 0] + dx[:, :, 1]**2 * sorted_conic[:, 1, 1]+ dx[:,:,0]*dx[:,:,1] * sorted_conic[:, 0, 1]+ dx[:,:,0]*dx[:,:,1] * sorted_conic[:, 1, 0]
    depth_acc = depths[None].expand_as(dist2)

    image, depthmap, opacity_map = alpha_blending_with_gaussians(dist2, colors, opacities, depth_acc, H, W)

    return {
            "render" : image.permute(2, 0, 1), 
            "depth"  : depthmap, 
            "opacity": opacity_map,
            "center" : means2D, 
            "radii"  : radii, 
            "dist2"  : dist2
        }