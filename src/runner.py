import numpy as np
import torch
from catalyst import dl

from .loss import CustomL2Loss
from .transforms import upscale_keypoints_torch


class CustomRunner(dl.SupervisedRunner):
    loss = CustomL2Loss()

    def _handle_batch(self, batch):
        y_pred = self.model(batch['features'])
        if len(batch['targets']):
            self.state.batch_metrics.update(
                {"mse": torch.pow(upscale_keypoints_torch(batch['targets'] - y_pred, batch["size"]), 2).mean(axis=1).mean()}
            )
        self.state.input = {"features": batch['features'], "targets": batch['targets'],
                            'sizes': batch["size"], 'filenames': batch["filename"]}
        self.state.output = {"logits": y_pred}
