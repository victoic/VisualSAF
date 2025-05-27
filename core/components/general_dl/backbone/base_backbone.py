import torch
from torch import nn
from torchvision import transforms

from ....base_model import BaseModule
from typing import List, Tuple

class BaseBackbone(BaseModule):    
    def get_channels(self, level: int | List[int] | Tuple[int] = None):
        if level is None:
            return self.channels
        if type(level) in (list, tuple):
            channels = []
            for i, channel in enumerate(self.channels):
                if i in level:
                    channels.append(channel)
                return tuple(channels)
        return self.channels[level]