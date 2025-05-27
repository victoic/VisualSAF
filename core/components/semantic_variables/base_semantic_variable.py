import torch
from torch import nn
from ...base_model import BaseModule
from abc import ABC, abstractmethod
from typing import List

class BaseSemanticVariable(BaseModule, ABC):
    def __init__(self, in_channels: tuple = (None, None, None, None), *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.is_input_img = False
        self.is_input_x = False
        self.is_input_backbone_feats = False
        self.is_input_det_output = False
        self.set_inputs()
        self.set_channels(in_channels)
    
    @abstractmethod
    def set_inputs(self):
        """
        Sets which inputs from the model should be arguments for the semantic variables.
        This must set the boolean attributes is_input_img, is_input_x, is_input_backbone_feats, 
        and is_input_det_output to True where needed.
        """
        pass
    
    def get_inputs(self, img, x, backbone_feats, det_output):
        inputs = []
        if self.is_input_img: inputs.append(img) 
        else: inputs.append(None)
        if self.is_input_x: inputs.append(x)
        else: inputs.append(None)
        if self.is_input_backbone_feats: inputs.append(backbone_feats)
        else: inputs.append(None)
        if self.is_input_det_output: inputs.append(det_output)
        else: inputs.append(None)
        return tuple(inputs)
    
    def set_channels(self, input_channels: tuple) -> None:
        img_ch, x_ch, bb_ch, det_ch = input_channels
        self.in_channels = []
        if self.is_input_img: self.in_channels.append(img_ch) 
        else: self.in_channels.append(None)
        if self.is_input_x: self.in_channels.append(x_ch) 
        else: self.in_channels.append(None)
        if self.is_input_backbone_feats: self.in_channels.append(bb_ch)
        else: self.in_channels.append(None)
        if self.is_input_det_output: self.in_channels.append(det_ch)
        else: self.in_channels.append(None)

    @abstractmethod
    def get_outputs(self) -> List[int]:
        """
        Resturns a list of output sizes this semantic variables returns on its forward.
        """
        pass

    @abstractmethod
    def freeze(self):
        """
        Resturns a list of output sizes this semantic variables returns on its forward.
        """
        pass