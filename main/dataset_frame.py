import os
import time

import cv2
import numpy as np

from torch.utils.data import Dataset
from einops import rearrange
import random
import torch

class Anime_frame_Dataset(Dataset):
    def __init__(
            self,
            video_root: str,
            width: int = 512,
            height: int = 512,
            n_sample_frames: int = 16,
    ):
        self.video_root = video_root
        self.video_path = []
        if os.path.isdir(video_root):
            for video_name in os.listdir(video_root):
                self.video_path.append(os.path.join(self.video_root, video_name))
                # self.prompt.append(video_root.split('/')[-1].replace('_', ' '))
        else:
            self.video_path.append(video_root)

        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames

    def __len__(self):
        return len(self.video_path)

    def __getitem__(self, index):
        fr = os.listdir(self.video_path[index])
        fr_rate = len(fr) // self.n_sample_frames
        select_rate = random.randint(1, fr_rate)

        try:
            start_idx = random.randint(0, len(fr)-self.n_sample_frames*select_rate)
        except Exception as e:
            print('error')
        sample_index = list(range(start_idx, len(fr), select_rate))[:self.n_sample_frames]

        video = None

        for i in sample_index:
            frame = cv2.imread(os.path.join(self.video_path[index],fr[i]))
            frame = cv2.resize(frame, (self.width, self.height))[:, :, ::-1]
            frame = torch.Tensor(frame.copy())[None]
            if video == None:
                video = frame
            else:
                video = torch.cat([video,frame],dim = 0)

        # video = rearrange(video, "f h w c -> f c h w")
        video = rearrange(video, "f h w c -> c f h w")

        if random.uniform(0, 1) > 0.5:
            video = torch.flip(video, dims=[3])
        if random.uniform(0, 1) > 0.5:
            video = torch.flip(video, dims=[1])
        example = {
            "video": (video / 127.5 - 1.0),
            'caption': "",
            'fps': 10
        }
        return example
