from generate_data import generate_data
import torch.multiprocessing as mp
from utils.config_utils import load_config
from utils.logging_utils import Log
from toy_slam import config_save_results
from toy_slam import SLAM
import wandb
import numpy as np 

mp.set_start_method("spawn")
save_dir            = None
mode                = "2DGS"
project_name        = "toy_MonoGS_track_uncertainty"
config              = load_config("./configs/mono/tum/toy_config.yml")
dataset_name        = "tmp_image"
file_name_gaussians = "tmp_gaussians.json"
num_perspectives    = 5

for high in np.linspace(0.2, 2, num_perspectives):
    center_eye = np.array([0, 0, high])
    generate_data(center_eye)

    if config["Results"]["save_results"]:
        save_dir = config_save_results(config, dataset_name=dataset_name, project_name=project_name)

    slam = SLAM(config, save_dir=save_dir, mode=mode, file_name_gaussians=file_name_gaussians)

    slam.run()
    wandb.finish()

    # All done
    Log("Done.")