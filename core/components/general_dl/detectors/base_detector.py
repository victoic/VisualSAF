import torch
from ....base_model import BaseModule
from abc import ABC, abstractmethod
from typing import List

class BaseDetector(BaseModule, ABC):
    def get_input(self, imgs, transformed_imgs, feats):
        """
        Decifed for specific Detector if input is image or backbone output features.
        Only needs to be overwritten in case the last level of the backbone features
        is not the desired input for the detector.
        Arguments:
        - img: input images, PIL.Image
        - transformed_imgs: tranformed input images, torch.Tensor
        - feats: bacbone outputs, List[torch.Tensor]
        Return:
        Either img, feats or a specific level of the features.
        """
        return feats[-1]

    @abstractmethod
    def get_channels(self) -> List[int]:
        """
        Returns output dimensions which might be necessary for future modules.
        Return:
        List of integers containing each important number of output channels.
        """
        pass