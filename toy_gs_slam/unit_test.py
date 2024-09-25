import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

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
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        return Image.fromarray(image)

    def generate_dummy_pose(self, idx):
        # Generate a dummy pose (e.g., identity or slight translation)
        pose = np.eye(4, dtype=np.float32)
        pose[0, 3] = idx * 0.1  # Translate along x-axis
        return pose
