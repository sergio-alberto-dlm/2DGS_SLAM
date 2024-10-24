#################################################
#   Script to generate artificial toy cameras   #
#################################################

import os 
import glob 
import time
import json  

import cv2
import torch 
import numpy as np 
from munch import munchify

import matplotlib.pyplot as plt 
import matplotlib

from toy_utils.utils_cameras import generate_circular_trajectory
from toy_utils.utils_cameras import generate_circular_trajectory_xy
from toy_utils.utils import toy_gaussian_model
from toy_gaussian_splatting.gaussian_2D_render import surface_splatting
from toy_gaussian_splatting.gaussian_3D_render import volume_splatting


def generate_data(center_eye : np.array, dir_image="toy_data/tmp_image", dir_depth="toy_data/tmp_depth"):
    DEVICE = 'cuda:0'
    MODE   = '2DGS'
    render_video = False 

    # create the directories to store the renders 
    def create_dir(name_dir : str):
        """
        Create the path to store the renders 
        """
        if not os.path.exists(name_dir):
            os.makedirs(name_dir)

            print(f'Render directory: {name_dir}')
        return name_dir

    #image_folder, depth_folder = create_dir("toy_data/rgb_circle"), create_dir("toy_data/depth_circle")
    image_folder, depth_folder = create_dir(dir_image), create_dir(dir_depth)

    # make inputs
    num_points1 = 8
    gaussians   = toy_gaussian_model(num_points=num_points1)
    gaussians_json = gaussians.copy()
    for key in gaussians_json:
        gaussians_json[key] = gaussians_json[key].cpu().tolist()
    # store gaussians 
    with open("toy_data/tmp_gaussians.json", "w", encoding="utf-8") as f:
            json.dump(gaussians_json, f, indent=4)
    gaussians = munchify(gaussians)
    height, width = 512, 512

    # set up cameras parameters 
    center         = center_eye
    radius         = 2.3
    num_steps      = 100
    #cameras_circle = generate_circular_trajectory(center, radius, num_steps)
    cameras_circle = generate_circular_trajectory_xy(center, radius, num_steps)

    poses = []

    start = time.time()
    if MODE == '2DGS':
        for idx, c2w in enumerate(cameras_circle):
            viewmat   = torch.linalg.inv(c2w)
            viewpoint = munchify(
                {"R" : viewmat[:3, :3], "T" : viewmat[:3, 3], 
                "cam_trans_delta": torch.zeros(3).to(DEVICE), "cam_rot_delta": torch.zeros(3).to(DEVICE)}
            )
            render2D = surface_splatting(
                gaussians.means3D, gaussians.scales, gaussians.quats, viewpoint, gaussians.projmat, 
                gaussians.colors, gaussians.opacities, gaussians.intrins, device=DEVICE, 
            )   

            image_np = render2D['render'].permute(1, 2, 0).cpu().numpy()
            depth_np = render2D['depth'].cpu().numpy().squeeze()
            poses.append(viewmat)

            # print(image_np.shape, depth_np.shape)

            plt.imsave(f"{image_folder}/frame_{idx:04d}.jpg", image_np)
            plt.imsave(f"{depth_folder}/frame_{idx:04d}.jpg", depth_np, cmap='viridis')

            #print(f"Rendered frame {idx+1}/{len(cameras_circle)}")

    elif MODE == '3DGS':

        for idx, c2w in enumerate(cameras_circle):
            viewmat   = torch.linalg.inv(c2w)
            viewpoint = munchify(
                {"R" : viewmat[:3, :3], "T" : viewmat[:3, 3]}
            )
            render3D = volume_splatting(
            gaussians.means3D, gaussians.scales, gaussians.quats, viewpoint, gaussians.projmat, 
            gaussians.colors, gaussians.opacities, gaussians.intrins, device=DEVICE, 
            )   

            image_np = render3D['render'].permute(1, 2, 0).cpu().numpy()
            depth_np = render3D['depth'].cpu().numpy().squeeze()
            poses.append(viewmat)

            plt.imsave(f"{image_folder}/frame_{idx:04d}.jpg", image_np)
            plt.imsave(f"{depth_folder}/frame_{idx:04d}.jpg", depth_np, cmap='viridis')

            #print(f"Rendered frame {idx+1}/{len(cameras_circle)}")

    print(f"Total time renderization: {time.time() - start}")

    # Open the file in write mode
    with open("toy_data/tmp_poses.txt", "w") as f:
        for pose in poses:
            # Flatten the matrix and join the values into a single line
            pose_line = ' '.join(map(str, pose.numpy().flatten()))
            f.write(pose_line + "\n")


    if render_video:

        image_files = glob.glob(os.path.join(image_folder, '*.png')) 
        depth_files = glob.glob(os.path.join(depth_folder, '*.png'))
        fps         = 3
        
        image_writer = cv2.VideoWriter(os.path.join(image_folder, 'rgb_circle.mp4'), cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))
        depth_writer = cv2.VideoWriter(os.path.join(depth_folder, 'depth_circle.mp4'), cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))

        # render image video 
        for file_image, file_depth in zip(image_files, depth_files):

            frame_image = cv2.imread(file_image)
            frame_depth = cv2.imread(file_depth)  
            
            image_writer.write(frame_image)
            depth_writer.write(frame_depth)

        image_writer.release()
        depth_writer.release()

        print("Finish to render video")
