"""Contains combination of supervisor and supdynamics."""
import torch.nn as nn

from .dynamics import SupervisedDynamics
from .supervisor import Supervisor


class SupStove(nn.Module):
    """Combination of classes for supervised approach."""

    def __init__(self, config):
        """Initialise combined class."""
        super().__init__()
        self.sup = Supervisor(config)
        self.dyn = SupervisedDynamics(config)
