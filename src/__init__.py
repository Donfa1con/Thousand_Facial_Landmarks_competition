from .runner import CustomRunner as Runner

from .experiment import Experiment
from .model import KeyPointNet
from .loss import CustomL2Loss
from .callbacks import KeypointsInferCallback
from catalyst.dl import registry

registry.Model(KeyPointNet)
registry.Criterion(CustomL2Loss)
registry.Callback(KeypointsInferCallback)
