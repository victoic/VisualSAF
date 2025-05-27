from torch import nn as nn

def add_channels_to_args(model: nn.Module, position: int, args: list = [], kwargs: dict = {}):
    channels = model.backbone.get_channels()
    args.insert(position, channels)

def add_num_feature_levels_to_args(model: nn.Module, position: int, args: list = [], kwargs: dict = {}):
    num_feature_levels = len(model.backbone.get_channels())
    args.insert(position, num_feature_levels)