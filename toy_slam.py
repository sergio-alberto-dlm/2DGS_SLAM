import os
import sys
import time
from argparse import ArgumentParser
from datetime import datetime

import torch
import torch.multiprocessing as mp
import yaml
from munch import munchify

import wandb
from toy_utils.utils import toy_gaussian_model #<-----------------------------------
from gaussian_splatting.utils.system_utils import mkdir_p
from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.eval_utils import eval_ate #eval_rendering, save_gaussians
from utils.logging_utils import Log
from toy_utils.toy_frontend import FrontEnd
from toy_utils.utils import load_gaussians

def config_save_results(config, dataset_name="toy_dataset_2D", project_name="Mono2DGS"):
    mkdir_p(config["Results"]["save_dir"])
    current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # path = config["Dataset"]["dataset_path"].split("/")
    save_dir = os.path.join(
        config["Results"]["save_dir"], dataset_name, current_datetime
    )
    # tmp = args.config
    tmp = dataset_name #tmp.split(".")[0]
    config["Results"]["save_dir"] = save_dir
    mkdir_p(save_dir)
    with open(os.path.join(save_dir, "config.yml"), "w") as file:
        documents = yaml.dump(config, file)
    Log("saving results in " + save_dir)
    run = wandb.init(
            project=project_name,
            name=f"{tmp}_{current_datetime}",
            config=config,
            mode=None if config["Results"]["use_wandb"] else "disabled",
        )
    wandb.define_metric("frame_idx")
    wandb.define_metric("ate*", step_metric="frame_idx")
    wandb.define_metric("avg_error_track", step_metric="frame_idx")
    wandb.define_metric("track_iters", step_metric="frame_idx")
    return save_dir

class SLAM:
    def __init__(self, config, save_dir=None, mode="2DGS", file_name_gaussians="gaussians.json"):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        self.config     = config
        self.save_dir   = save_dir
        model_params    = munchify(config["model_params"])
        opt_params      = munchify(config["opt_params"])
        pipeline_params = munchify(config["pipeline_params"])
        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,
            opt_params,
            pipeline_params,
        )

        self.monocular      = self.config["Dataset"]["sensor_type"] == "monocular"
        self.eval_rendering = self.config["Results"]["eval_rendering"]
        self.dataset        = load_dataset(
            model_params, model_params.source_path, config=config
        )
        frontend_queue = mp.Queue()


        self.config["Results"]["save_dir"]   = save_dir
        self.config["Training"]["monocular"] = self.monocular

        self.frontend                = FrontEnd(self.config)
        self.frontend.gaussians      = munchify(load_gaussians(config, file_name_gaussians=file_name_gaussians))
        self.frontend.dataset        = self.dataset
        self.frontend.frontend_queue = frontend_queue
        self.frontend.device         = "cuda:0" 
        self.frontend.mode           = mode
        self.frontend.set_hyperparams()


        self.frontend.run()
        end.record()

        # empty the frontend queue
        N_frames = len(self.frontend.cameras)
        FPS = N_frames / (start.elapsed_time(end) * 0.001)
        Log("Total time", (start.elapsed_time(end) * 0.001), tag="Eval")
        Log("Total FPS", N_frames / (start.elapsed_time(end) * 0.001), tag="Eval")

        if self.eval_rendering:
            self.gaussians = self.frontend.gaussians
            try:
                ATE = eval_ate(
                    self.frontend.cameras,
                    range(len(self.dataset)), #self.frontend.kf_indices,
                    self.save_dir,
                    0,
                    final=True,
                    monocular=self.monocular,
                )

                columns = ["RMSE ATE", "FPS"]
                metrics_table = wandb.Table(columns=columns)
                metrics_table.add_data(
                    ATE,
                    FPS,
                )
                wandb.log({"Metrics": metrics_table})
            except Exception as e:
                print(f"An error occurred: {e}")

    def run(self):
        pass


if __name__ == "__main__":
    mp.set_start_method("spawn")
    save_dir = None
    mode     = "2DGS"
    config   = load_config("./configs/mono/tum/toy_config.yml")
    dataset_name        = "toy_dataset_2D"
    project_name        = "MonoGS"
    file_name_gaussians = "gaussians.json"

    if config["Results"]["save_results"]:
        save_dir = config_save_results(config, dataset_name=dataset_name, project_name=project_name)

    slam = SLAM(config, save_dir=save_dir, mode=mode, file_name_gaussian=file_name_gaussians)

    slam.run()
    wandb.finish()

    # All done
    Log("Done.")
