import torch 

DEVICE = 'cpu'

def alpha_blending(alpha, colors):
    T = torch.cat([torch.ones_like(alpha[-1:]), (1-alpha).cumprod(dim=0)[:-1]], dim=0)
    image = (T * alpha * colors).sum(dim=0).reshape(-1, colors.shape[-1])
    alphamap = (T * alpha).sum(dim=0).reshape(-1, 1)
    return image, alphamap

def alpha_blending_with_gaussians(dist2, colors, opacities, depth_acc, H, W):
    colors = colors.reshape(-1,1,colors.shape[-1])
    depth_acc = depth_acc.T[..., None]
    depth_acc = depth_acc.repeat(1,1,1)

    # evaluate gaussians
    # just for visualization, the actual cut off can be 3 sigma!
    cutoff = 1**2
    dist2 = dist2.T
    gaussians = torch.exp(-0.5*dist2) * (dist2 < cutoff)
    gaussians = gaussians[..., None]
    alpha = opacities.unsqueeze(1) * gaussians

    # accumulate gaussians
    image, _ = alpha_blending(alpha, colors)
    depthmap, alphamap = alpha_blending(alpha, depth_acc)
    depthmap = depthmap / alphamap
    depthmap = torch.nan_to_num(depthmap, 0, 0)
    return image.reshape(H,W,-1), depthmap.reshape(H,W,-1)
