#################################################
#   Script to generate artificial toy cameras   #
#   i.e to generate                             #
#                                               #
#################################################

import os 
import glob 
import time 

import cv2
import torch 
import numpy as np 

import matplotlib.pyplot as plt 
import matplotlib

from tiny_renderer.utils_cameras import generate_circular_trajectory
from tiny_renderer.utils_cameras import get_inputs
from tiny_renderer.gaussian_2D_render import GaussianSplatRenderer2D

DEVICE = 'cpu'

# create the directories to store the renders 
def create_dir(name_dir : str):
    """
    Create the path to store the renders 
    """
    if not os.path.exists(name_dir):
        os.makedirs(name_dir)

        print(f'Render directory: {name_dir}')
    return name_dir

image_folder, depth_folder = create_dir("rgb_circle"), create_dir("depth_circle")
render_video = False # True 

# make inputs
num_points1=8
means3D, scales, quats = get_inputs(num_points=num_points1)
colors  = matplotlib.colormaps['Accent'](np.random.randint(1,64, 64)/64)[..., :3]
colors  = torch.from_numpy(colors).to(DEVICE)
opacity = torch.ones_like(means3D[:,:1])
height, width = 512, 512

# set up cameras parameters 
center         = np.array([0, 0, 0])
radius         = 3
num_steps      = 50
cameras_circle = generate_circular_trajectory(center, radius, num_steps)

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

start = time.time()
for idx, c2w in enumerate(cameras_circle):
    viewmat = torch.linalg.inv(c2w).permute(1,0)

    gaussian_2D = GaussianSplatRenderer2D()
    gaussian_2D.means3D   = means3D
    gaussian_2D.scales    = scales
    gaussian_2D.quats     = quats
    gaussian_2D.opacities = opacity
    gaussian_2D.colors    = colors
    gaussian_2D.viewmat   = viewmat
    gaussian_2D.projmat   = projmat
    gaussian_2D.intrins   = intrins

    image, depthmap, center, radii, dist = gaussian_2D.surface_splatting()

    image_np = image.cpu().numpy()
    depth_np = depthmap.cpu().numpy().squeeze()

    plt.imsave(f"{image_folder}/frame_{idx:04d}.png", image_np)
    plt.imsave(f"{depth_folder}/frame_{idx:04d}.png", depth_np, cmap='viridis')

    print(f"Rendered frame {idx+1}/{len(cameras_circle)}")

print(f"Total time renderization: {start - time.time()}")

if render_video:

    image_files = glob.glob(os.path.join(image_folder, '*.png')) 
    depth_files = glob.glob(os.path.join(depth_folder, '*.png'))
    fps         = 3
    
    image_writer = cv2.VideoWriter("./rgb_circle/rgb_circle.mp4", cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))
    depth_writer = cv2.VideoWriter("./depth_circle/depth_circle.mp4", cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))

    # render image video 
    for file_image, file_depth in zip(image_files, depth_files):

        frame_image = cv2.imread(file_image)
        frame_depth = cv2.imread(file_depth)  
        
        image_writer.write(frame_image)
        depth_writer.write(frame_depth)

    image_writer.release()
    depth_writer.release()

    print("Finish to render video")
