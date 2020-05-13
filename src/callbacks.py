import json

import pandas as pd
from catalyst.dl import Callback, CallbackOrder

from .transforms import upscale_keypoints_torch


class KeypointsInferCallback(Callback):

    def __init__(self, subm_file, test_points_csv, out_file):
        super().__init__(CallbackOrder.Internal)
        self.out_file = out_file
        self.subm = pd.read_csv(subm_file, index_col='file_name')
        self.test_points = pd.read_csv(test_points_csv, sep='\t', index_col='file_name')
        self.test_points['point_index_list'] = self.test_points.point_index_list.apply(lambda x: json.loads(x))

    def on_batch_end(self, state):
        batch_coords = upscale_keypoints_torch(state.output["logits"], state.input["sizes"]).detach().cpu().numpy()
        for coords, filename in zip(batch_coords, state.input['filenames']):
            points = []
            for point in self.test_points.loc[filename]['point_index_list']:
                points.append(point * 2)
                points.append(point * 2 + 1)
            self.subm.loc[filename] = coords[points]

    def on_loader_end(self, _):
        self.subm[self.subm.columns].astype(int).reset_index().to_csv(self.out_file, index=False)
