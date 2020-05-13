import random

import torch

from .constants import FLIPPED_IDXS, NUM_PTS


def upscale_keypoints(keypoints, img_size):
    return keypoints * (img_size * NUM_PTS)


def upscale_keypoints_torch(keypoints, sizes):
    return keypoints * torch.stack(sizes, dim=1).repeat(1, NUM_PTS)


def downscale_keypoints(keypoints, img_size):
    return keypoints / (img_size * NUM_PTS)


class HorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, keypoints):
        if random.random() < self.p:
            keypoints = keypoints.reshape(-1, 2)[FLIPPED_IDXS].ravel()
            keypoints[range(0, len(keypoints), 2)] = 1 - keypoints[range(0, len(keypoints), 2)]
            image = image[:, ::-1]
        return image, keypoints
