import torch
from torch import nn
import json, os, importlib

from core.base_model import BaseModel
from core.util import make_arg

"""
Builder class for loading configurations file and create semantic model.
"""
class Builder:
  """
  This is the builder class for loading semantic model from config file.
  """
  def __init__(self, config: str | dict):
    """
    Initilize the builder.
    args: 
      config (str): path to configuration file.
    """
    self.config = config
  
  @staticmethod
  def load_config(config: str) -> dict:
    """
    Loads configuration file into dictionary.
    args: 
      config (str): path to configuration file.
    """
    #assert os.path.isfile(config), f"Configuration file not found in {config}"
    #with open(config, 'r') as json_file:
    #  config = json.load(json_file)
    config = getattr(importlib.import_module(f'configs.{config}'), 'config')
    return config
  
  def get_datasets(self, device):
    datasets = {
      "train": None,
      "eval": None,
      "test": None
    }
    for dataset in self.config['datasets']:
      ds = dataset['dataset']
      ds_pck = 'datasets.' + ds['pck']
      dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), ds_pck.replace('.', '/'))
      ds_cls = ds['cls']
      cls = getattr(importlib.import_module(ds_pck), ds_cls)
      creator = cls
      if (ds['builder'] is not None):
        creator = getattr(cls, ds['builder'])
      args = make_arg(ds['args'])
      kwargs = ds['kwargs']
      new_ds = creator(*args, **kwargs)
      if (datasets[ds['type']] is None):
        datasets[ds['type']] = new_ds
      elif (type(datasets[ds['type']]) == list):
        datasets[ds['type']].append(new_ds)
      else:
        datasets[ds['type']] = [datasets[ds['type']]]
        datasets[ds['type']].append()
    return datasets
      

  def get_model(self, device) -> nn.Module:
    """
    Creates BaseModel module from configuration.
    return:
      BaseModel
    """
    model = BaseModel(self.config['model']['hooks'], device)
    model.bs = self.config['trainer']['batch_size']
    self.get_backbone(model)
    self.get_detector(model)
    #self.get_semantic_variables(model)
    #self.get_output_head(model)
    return model

  def get_backbone(self, model: BaseModel) -> None:
    backbone = self.config['backbone']
    model.set_backbone(backbone)

  def get_detector(self, model: BaseModel) -> None:
    detector = self.config['detector']
    model.set_detector(detector)

  def get_semantic_variables(self, model: BaseModel) -> None:
    semantic_variables = self.config['semantic_variables']
    model.set_semantic_variables(semantic_variables)
  
  def get_output_head(self, model: BaseModel) -> None:
    output_head = self.config['output_head']
    model.set_output_head(output_head)