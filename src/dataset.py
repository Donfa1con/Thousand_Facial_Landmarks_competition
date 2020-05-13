import os

import cv2
from torch.utils.data import Dataset

from .transforms import HorizontalFlip, downscale_keypoints


class KeyPointsDataset(Dataset):

    def __init__(self, data_df, size_df, img_dir, mode: str, transforms=None):
        assert mode in ["train", "valid", "infer"], "Mode should be one of `train`, `valid` or `infer`"
        self.data_df = data_df
        self.size_df = size_df
        self.img_dir = img_dir
        self.mode = mode
        self.transforms = transforms
        self.fliptransform = HorizontalFlip(0.5)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        item = {}
        img_data = self.data_df.iloc[idx]
        img = cv2.imread(os.path.join(self.img_dir, img_data.name))[..., ::-1]
        keypoints = []
        size = self.size_df.loc[img_data.name].values[::-1].tolist()
        if self.mode != "infer":
            keypoints = downscale_keypoints(img_data.values, size)
            img, keypoints = self.fliptransform(img, keypoints)
        img = self.transforms(image=img)["image"]
        item['size'] = size
        item['filename'] = img_data.name
        item['features'] = img
        item['targets'] = keypoints
        return item
