import os
from collections import OrderedDict

import albumentations as A
from albumentations.pytorch import ToTensor
import numpy as np
import pandas as pd
from catalyst.dl import ConfigExperiment
from sklearn.model_selection import train_test_split

from .dataset import KeyPointsDataset


class Experiment(ConfigExperiment):
    def get_datasets(self, stage: str, **kwargs):
        datasets = OrderedDict()
        data_params = self.stages_config[stage]["data_params"]

        orig_sizes = pd.read_csv(os.path.join(data_params["data_dir"], 'orig_size.csv'), index_col='file_name')
        if stage != "infer":
            landmarks = os.path.join(data_params["data_dir"], 'train/landmarks.csv')
            dtypes = {col: np.int16 for col in pd.read_csv(landmarks, sep='\t', index_col='file_name', nrows=1).columns}
            train_landmarks = pd.read_csv(landmarks, sep='\t', index_col='file_name', dtype=dtypes)
            train, valid = train_test_split(train_landmarks, test_size=data_params["valid_size"], random_state=42)

            datasets["train"] = KeyPointsDataset(data_df=train,
                                                 size_df=orig_sizes.loc[train.index],
                                                 img_dir=os.path.join(data_params["data_dir"], 'train/images'),
                                                 mode="train",
                                                 transforms=self.get_transforms())

            datasets["valid"] = KeyPointsDataset(data_df=valid,
                                                 size_df=orig_sizes.loc[valid.index],
                                                 img_dir=os.path.join(data_params["data_dir"], 'train/images'),
                                                 mode="valid",
                                                 transforms=self.get_test_transforms())
        else:
            subm = pd.read_csv(os.path.join(data_params["data_dir"], 'sampleSubmission.csv'), index_col='file_name')
            datasets["infer"] = KeyPointsDataset(data_df=subm,
                                                 size_df=orig_sizes.loc[subm.index],
                                                 img_dir=os.path.join(data_params["data_dir"], 'test/images'),
                                                 mode="infer",
                                                 transforms=self.get_test_transforms())
        return datasets

    @staticmethod
    def get_transforms():
        return A.Compose([
            A.MotionBlur(p=0.2),
            A.OneOf([A.HueSaturationValue(p=0.5),
                     A.RGBShift(p=0.5)], p=1),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(),
            ToTensor()
        ])

    @staticmethod
    def get_test_transforms():
        return A.Compose([A.Normalize(), ToTensor()])
