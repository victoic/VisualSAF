import torch, numpy as np
from typing import List
from ...base_model import BaseModule
from abc import ABC, abstractmethod
from collections import OrderedDict

class BaseOutputHead(BaseModule, ABC):
    def __init__(self, semantic_sizes: np.ndarray, img_size: List[int], num_classes: int, 
                 channels: List[int], use_img: List[bool],  *args, **kwargs) -> None:
        
        super(BaseOutputHead, self).__init__(*args, **kwargs)
        
        self.reducers = []
        counter = 0
        for so in semantic_sizes:
            dim = so
            if (type(so) == np.ndarray):
                dim = np.prod(so)
            self.reducers.append(
                torch.nn.Linear(dim, channels[0]),
            )
            counter+=1
        self.reducers = torch.nn.ModuleList(self.reducers)
        
        layers = [torch.nn.Linear(counter*channels[0], channels[1])]
        if len(channels) > 2:
            for channel in range(2, len(channels)):
                layers.append(torch.nn.Linear(channels[channel-1], channels[channel]))
        self.model = torch.nn.Sequential(*layers)
        self.final_output = torch.nn.Linear(channels[-1], num_classes)
        self.activation = torch.nn.Sigmoid() 
    
    def forward(self, semantic_outputs: list):
        outs = []
        for i in range(len(semantic_outputs)):
            ins: torch.Tensor = semantic_outputs[i]
            if (len(ins.shape) > 3):
                ins = ins.reshape((ins.shape[0], np.prod(ins.shape[1:])))
            if self.reducers[i].in_features != ins.shape[-1]:
                in_features = self.reducers[i].in_features
                end_index = min(in_features, np.prod(ins.shape[1:]))
                _flat = torch.zeros((ins.shape[0], in_features)
                                   , device=semantic_outputs[i].device, dtype=semantic_outputs[i].dtype)
                for img in range(ins.shape[0]):
                    _flat[img, :end_index] = ins[img].flatten()[:end_index]
                ins = _flat
            ins = ins.type(torch.float)
            out = self.reducers[i](ins)
            outs.append(out)

        outs = tuple(outs)   
        det_outs = torch.cat((outs[0], outs[1]), -1)
        out2_shape = (outs[2].shape[0], outs[0].shape[1], outs[2].shape[1])       
        out3_shape = (outs[3].shape[0], outs[0].shape[1], outs[3].shape[1])       
        outs = torch.cat((outs[0],
                          outs[1],
                          outs[2].repeat(1, 1, outs[0].shape[1]).reshape(out2_shape), 
                          outs[3].repeat(1, 1, outs[0].shape[1]).reshape(out3_shape)), -1)
        out = self.model(outs)
        out = self.final_output(out)
        return self.activation(out)

    @abstractmethod
    def set_inputs(self):
        pass

