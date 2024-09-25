import torch 

from tiny_renderer.utils import build_scaling_rotation, homogeneous, homogeneous_vec
from tiny_renderer.utils_render import alpha_blending_with_gaussians

DEVICE = 'cpu'

class GaussianSplatRenderer3D:

    def __init__(self, device="cpu"):
        self.device    = device
        self.means3D   = None
        self.cov3d     = None 
        self.cov2d     = None 
        self.scales    = None
        self.quats     = None
        self.opacities = None
        self.colors    = None
        self.viewmat   = None
        self.projmat   = None
        self.intrins   = None

    def build_covariance_2d(self, tan_fovx, tan_fovy, focal_x, focal_y):
        import math
        t = (self.means3D @ self.viewmat[:3,:3]) + self.viewmat[-1:,:3]
        tz = t[..., 2]
        tx = t[..., 0]
        ty = t[..., 1]

        # Eq.29 locally affine transform
        # perspective transform is not affine so we approximate with first-order taylor expansion
        # notice that we multiply by the intrinsic so that the variance is at the sceen space
        J = torch.zeros(self.means3D.shape[0], 3, 3).to(self.means3D)
        J[..., 0, 0] = 1 / tz * focal_x
        J[..., 0, 2] = -tx / (tz * tz) * focal_x
        J[..., 1, 1] = 1 / tz * focal_y
        J[..., 1, 2] = -ty / (tz * tz) * focal_y
        W = self.viewmat[:3,:3].T # transpose to correct viewmatrix
        self.cov2d = J @ W @ self.cov3d @ W.T @ J.permute(0,2,1)

        # add low pass filter here according to E.q. 32
        filter = torch.eye(2,2).to(self.cov2d) * 0.0
        self.cov2d = self.cov2d[:, :2, :2] + filter[None]
        return 

    def build_covariance_3d(self):
        L = build_scaling_rotation(self.scales, self.quats).permute(0,2,1)
        self.cov3d = L @ L.transpose(1, 2)
        return 

    def projection_ndc(self):
        points_o = homogeneous(self.means3D) # object space
        points_h = points_o @ self.viewmat @ self.projmat # screen space # RHS
        p_w = 1.0 / (points_h[..., -1:] + 0.000001)
        p_proj = points_h * p_w
        p_view = points_o @ self.viewmat
        in_mask = p_view[..., 2] >= 0.2
        return p_proj, p_view, in_mask

    def get_radius(self):
        det = self.cov2d[:, 0, 0] * self.cov2d[:,1,1] - self.cov2d[:, 0, 1] * self.cov2d[:,1,0]
        mid = 0.5 * (self.cov2d[:, 0,0] + self.cov2d[:,1,1])
        lambda1 = mid + torch.sqrt((mid**2-det).clip(min=0.1))
        lambda2 = mid - torch.sqrt((mid**2-det).clip(min=0.1))
        return 3.0 * torch.sqrt(torch.max(lambda1, lambda2)).ceil()

    def volume_splatting(self):
        self.projmat = torch.zeros(4,4).to(DEVICE)
        self.projmat[:3,:3] = self.intrins
        self.projmat[-1,-2] = 1.0
        self.projmat = self.projmat.T

        mean_ndc, mean_view, in_mask = self.projection_ndc()

        depths = mean_view[:,2]
        mean_coord_x = mean_ndc[..., 0]
        mean_coord_y = mean_ndc[..., 1]

        means2D = torch.stack([mean_coord_x, mean_coord_y], dim=-1)
        # scales = torch.cat([scales[..., :2], scales[..., -1:]*1e-2], dim=-1)
        self.build_covariance_3d()

        W, H = (self.intrins[0,-1] * 2).long().item(), (self.intrins[1,-1] * 2).long().item()
        fx, fy = self.intrins[0,0], self.intrins[1,1]
        tan_fovx = W / (2 * fx)
        tan_fovy = H / (2 * fy)
        self.build_covariance_2d(tan_fovx, tan_fovy, fx, fy)
        radii = self.get_radius()

        # Rasterization
        # generate pixels
        pix = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W), indexing='xy'), dim=-1).to(DEVICE).flatten(0,-2)
        sorted_conic = self.cov2d.inverse() # inverse of variance
        dx = (pix[:,None,:] - means2D[None,:]) # B P 2
        dist2 = dx[:, :, 0]**2 * sorted_conic[:, 0, 0] + dx[:, :, 1]**2 * sorted_conic[:, 1, 1]+ dx[:,:,0]*dx[:,:,1] * sorted_conic[:, 0, 1]+ dx[:,:,0]*dx[:,:,1] * sorted_conic[:, 1, 0]
        depth_acc = depths[None].expand_as(dist2)

        image, depthmap = alpha_blending_with_gaussians(dist2, self.colors, self.opacities, depth_acc, H, W)
        return image, depthmap, means2D, radii, dist2

        
