import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch.multiprocessing as mp 

from toy_utils.toy_frontend import FrontEnd
from utils.config_utils import load_config
from gaussian_splatting.utils.graphics_utils import focal2fov


class MockDataset(Dataset):
    def __init__(self, num_frames, image_size=(640, 480)):
        self.num_frames = num_frames
        self.image_size = image_size
        self.fx = 525.0  # Focal length in x
        self.fy = 525.0  # Focal length in y
        self.cx = image_size[0] / 2.0  # Principal point x
        self.cy = image_size[1] / 2.0  # Principal point y
        self.width = image_size[0]
        self.height = image_size[1]
        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        
        # Generate dummy images and poses
        self.images = [self.generate_dummy_image() for _ in range(num_frames)]
        self.poses = [self.generate_dummy_pose(i) for i in range(num_frames)]
        self.device = "cpu"

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        image = self.images[idx]
        depth = None  # If you have depth maps, include them here
        pose = self.poses[idx]
        return image, depth, pose

    def generate_dummy_image(self):
        # Generate a dummy image (e.g., a blank image or a simple pattern)
        image = torch.zeros((3, self.height, self.width), dtype=torch.float32)
        return image

    def generate_dummy_pose(self, idx):
        # Generate a dummy pose (e.g., identity or slight translation)
        pose = torch.eye(4, dtype=torch.float32)
        pose[0, 3] = idx * 0.1  # Translate along x-axis
        return torch.linalg.inv(pose)

class Gaussians:
    def __init__(self, device="cpu"):
        # Create dummy Gaussian parameters
        num_gaussians = 100  # Adjust as needed
        self.means3D = torch.rand(num_gaussians, 3, device=device)
        self.scales = torch.rand(num_gaussians, 3, device=device)
        self.quats = torch.rand(num_gaussians, 4, device=device)
        self.colors = torch.rand(num_gaussians, 3, device=device)
        self.opacities = torch.rand(num_gaussians, 1, device=device)
        self.projmat = torch.eye(4, device=device)
        self.intrins = torch.tensor([
            [525.0, 0.0, 320],
            [0.0, 525.0, 240],
            [0.0, 0.0, 1.0]
        ], device=device)

def test_frontend_tracking(config):
    # Initialize device
    device = torch.device("cpu")  # Use "cuda:0" if GPU is available and desired

    # Initialize 
    num_frames     = 5  # Number of frames to test
    dataset        = MockDataset(num_frames)
    gaussians      = Gaussians(device=device)
    frontend_queue = mp.Queue()
    

    # Initialize FrontEnd
    frontend                = FrontEnd(config)
    frontend.frontend_queue = frontend_queue
    frontend.gaussians      = gaussians
    frontend.device         = device
    frontend.dataset        = dataset 

    # Set hyperparameters (ensure this method is called)
    frontend.set_hyperparams()
    print("Frontend initialized")

    # Run the FrontEnd's run method
    frontend.run()

    # Check the outputs
    for idx in range(num_frames):
        if idx in frontend.cameras:
            camera = frontend.cameras[idx]
            print(f"Frame {idx}: Pose:\n{camera.R}\n{camera.T}")
        else:
            print(f"Frame {idx}: No camera data available.")

    # Additional checks can be performed here
    # For example, verify that the poses have been updated from initial values

config = load_config("./configs/mono/tum/toy_config.yml")
test_frontend_tracking(config)

