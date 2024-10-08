import time
import os 

import numpy as np
import torch
import torch.multiprocessing as mp
import wandb

from toy_gaussian_splatting.gaussian_2D_render import surface_splatting
from toy_gaussian_splatting.gaussian_3D_render import volume_splatting
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from utils.camera_utils import Camera
from utils.eval_utils import eval_ate, save_gaussians
from utils.logging_utils import Log
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_tracking, get_median_depth
from toy_utils.utils import log_image_table
from gaussian_splatting.utils.system_utils import mkdir_p

import matplotlib.pyplot as plt 

class FrontEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config             = config
        self.frontend_queue     = None
        self.monocular          = config["Training"]["monocular"]
        self.reset              = True
        self.use_every_n_frames = 1
        self.gaussians          = None   #<-------------------- here I am going to store the gaussian params for the tiny parameter 
        self.cameras            = dict()
        self.device             = "cuda:0"

        self.dataset = None 
        self.mode    = None
        self.shift_data = 0

    def set_hyperparams(self):
        self.save_dir         = self.config["Results"]["save_dir"]
        self.save_results     = self.config["Results"]["save_results"]
        self.save_trj         = self.config["Results"]["save_trj"]
        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]

    def initialize(self, cur_frame_idx, viewpoint):
        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)
        self.reset = False

    def tracking(self, cur_frame_idx, viewpoint):
        prev = self.cameras[cur_frame_idx - self.use_every_n_frames]
        viewpoint.update_RT(prev.R, prev.T)

        opt_params = []
        opt_params.append(
            {
                "params": [viewpoint.cam_rot_delta],
                "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                "name": "rot_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.cam_trans_delta],
                "lr": self.config["Training"]["lr"]["cam_trans_delta"],
                "name": "trans_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_a],
                "lr": 0.01,
                "name": "exposure_a_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_b],
                "lr": 0.01,
                "name": "exposure_b_{}".format(viewpoint.uid),
            }
        )

        pose_optimizer = torch.optim.Adam(opt_params)

        losses = []
        for tracking_itr in range(self.tracking_itr_num):
#------------------------------------------------------------------->
            if self.mode == '2DGS':
                render = surface_splatting(
                    self.gaussians.means3D, self.gaussians.scales, 
                    self.gaussians.quats, viewpoint, 
                    self.gaussians.projmat, self.gaussians.colors, 
                    self.gaussians.opacities, self.gaussians.intrins, 
                    device=self.device
                )
            elif self.mode == '3DGS':
                render = volume_splatting(
                    self.gaussians.means3D, self.gaussians.scales, 
                    self.gaussians.quats, viewpoint, 
                    self.gaussians.projmat, self.gaussians.colors, 
                    self.gaussians.opacities, self.gaussians.intrins, 
                    device=self.device
                )

            image, depth, opacity = (
                render["render"], 
                render["depth"], 
                render["opacity"]
            )
#------------------------------------------------------------------->
            pose_optimizer.zero_grad()
            loss_tracking = get_loss_tracking(
                self.config, image, depth, opacity.permute(2, 0, 1), viewpoint
            )
            losses.append(loss_tracking.item())

            loss_tracking.backward()
      
            with torch.no_grad():
                pose_optimizer.step()
                converged = update_pose(viewpoint)




            if converged:
                break

        avg_loss = np.array(losses).mean()
        wandb.log({"avg_track_error": avg_loss, "track_iters": tracking_itr})
        self.median_depth = get_median_depth(depth, opacity)
        return render


    def run(self):
        cur_frame_idx = 0
        projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.dataset.fx,
            fy=self.dataset.fy,
            cx=self.dataset.cx,
            cy=self.dataset.cy,
            W=self.dataset.width,
            H=self.dataset.height,
        ).transpose(0, 1)
        projection_matrix = projection_matrix.to(device=self.device)
        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)
        tmp_renders = {"image" : [], "ate" : []}

        for cur_frame_idx in np.array(range(len(self.dataset))) - self.shift_data:
            
            relative_idx = cur_frame_idx + self.shift_data
            if self.frontend_queue.empty():
                tic.record()

                viewpoint = Camera.init_from_dataset(
                    self.dataset, cur_frame_idx, projection_matrix
                )
                viewpoint.compute_grad_mask(self.config)

                self.cameras[relative_idx] = viewpoint

                if self.reset:
                    self.initialize(cur_frame_idx, viewpoint)
                    cur_frame_idx += 1
                    continue

                #print(viewpoint.R, viewpoint.T)

                # Tracking
                render = self.tracking(relative_idx, viewpoint)
                print(f"Track complete for frame: {relative_idx}")
                cur_frame_idx += 1

                if (
                    (relative_idx) % 20 == 0
                ):
                    Log("Evaluating ATE at frame: ", relative_idx)
                    try:
                        ate = eval_ate(
                            self.cameras,
                            range(relative_idx), #self.kf_indices,
                            self.save_dir,
                            relative_idx,
                            monocular=self.monocular,
                        )
                        tmp_renders["image"].append(render["render"].detach().permute(1, 2, 0).cpu().numpy())
                        tmp_renders["ate"].append(ate)
                        path_tmp_frame = os.path.join(self.config["Results"]["save_dir"], "tmp_frames")
                        mkdir_p(path_tmp_frame)
                        plt.imsave(os.path.join(path_tmp_frame, f"frame_{relative_idx:04d}.jpg"), render["render"].detach().permute(1, 2, 0).cpu().numpy())
                    except Exception as e:
                        print(f"An error occurred: {e}")
                toc.record()
                # torch.cuda.synchronize()
            else:
                print("Error with the fronend queue")
        
        log_image_table(tmp_renders)

        if self.save_results:
            eval_ate(
                self.cameras,
                range(len(self.dataset)), #self.kf_indices, <----------------------
                self.save_dir,
                0,
                final=True,
                monocular=self.monocular,
            )
        print("Fronend process finish")