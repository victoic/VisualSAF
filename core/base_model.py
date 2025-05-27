import importlib, os, numpy as np
import torch
from torch import nn
import torchvision.transforms as transforms
from .util import make_arg

from typing import Tuple, List, Dict

class BaseModule(nn.Module):
  def __init__(self, hooks: dict, device: str, preinit_return: tuple = None, load_from: str = None, base_dir: str = None) -> None:
    assert type(hooks) == dict, f"Wrong format for hooks, expected dict, got {type(hooks)}"
    super().__init__()
    self.device = device
    self.base_dir = base_dir
    self.load_from = load_from
    if (self.load_from != None):
      self.load_from = os.path.join('model-weights', self.load_from)
      print(f'Loading {self.__class__.__name__} model from {self.load_from}')
    self.init_hooks = hooks['init_hooks']
    self.preprocess_hooks = hooks['preprocess_hooks']
    self.postprocess_hooks = hooks['postprocess_hooks']
    #self.to(self.device)
    self.call_hooks('init_hooks')

  def get_hook_args(self, hook_dict: dict, is_kwargs: bool=False) -> List | Dict:
    hook_args = [] if not is_kwargs else {}
    for key, args in hook_dict.items():
      value = args['value']
      if args['type'] == 'attr':
        value = getattr(self, value)
      if not is_kwargs:
        hook_args.insert(args['index'], value)
      else:
        hook_args[key] = value
    return hook_args

  def call_hooks(self, hook_key: str, args: tuple = (None, None), kwargs: dict = {}, index: int = -1):
    return_values = args
    if hasattr(self, hook_key) and len(getattr(self, hook_key)) > 0:
      return_values = []
      attribute = getattr(self, hook_key) 
      if type(getattr(self, hook_key)) != list: attribute = attribute[index]
      for hook_dict in getattr(self, hook_key):
        pkg = hook_dict['package']
        mod = hook_dict['module']
        imp = f'core.components.{pkg}.hooks.{mod}'
        func = getattr(importlib.import_module(imp), hook_dict['hook'])

        hook_args = self.get_hook_args(hook_dict['args'])
        hook_kwargs = self.get_hook_args(hook_dict['kwargs'], is_kwargs=True)  
        for k, v in kwargs.items():
          hook_kwargs[k] = v

        return_val = func(self, args, *hook_args, **hook_kwargs)
        return_values = [*return_values, *return_val]
      return_values = tuple(return_values)
    return return_values
  
  def load_weight(self, state_path: str):
    print(f'Loading weight for {type(self).__name__} from {state_path}')
    state = torch.load(state_path)
    return state

class BaseModel(BaseModule):
  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.backbone: BaseModule = None
    self.detector: BaseModule = None
    self.semantic_variables: list = list()
    self.output: BaseModule = None
    self.img_channels = 3
    self.img_height = 224
    self.img_width = 224

    self.bs = 1

  def transform_images(self, imgs):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    imgs = transform(imgs)
    img_tensor = imgs
    if len(img_tensor[0].shape) < 4: 
        img_tensor = torch.unsqueeze(img_tensor, 0).to(self.device, dtype=torch.float32)
    return img_tensor

  def forward(self, img, img_metas=None) -> torch.Tensor:
    """ x = self.backbone.call_hooks('preprocess_hooks', args=(img, None))
    if type(x) == tuple and len(x) > 1:
      x, others_bb = x[0], x[1:]
    else:
      x = x[0]
      others_bb = None
    y, feats = self.backbone(x, others_bb)
    y, feats = self.backbone.call_hooks('postprocess_hooks', args=(y, feats)) """
    x = img
    feats = img
    
    x_det = self.detector.get_inputs(img, x, feats)
    x_det = self.detector.call_hooks('preprocess_hooks', args=(x_det, feats))
    if type(x_det) == tuple and len(x_det) > 1:
      x_det, others_det = x_det[0], x_det[1:]
    else:
      x_det = x_det[0]
      others_det = None
    out_det  = self.detector(x_det, feats, others_det)
    bbox_pred, cls_pred, others = self.detector.call_hooks('postprocess_hooks', args=(out_det, others_det))
    """ outputs = [bbox_pred, cls_pred]
    for s_var in self.semantic_variables:
      x_sv = s_var.get_inputs(img, x, feats, (cls_pred, bbox_pred))
      x_sv = s_var.call_hooks('preprocess_hooks', args=(x_sv, None))
      if type(x_sv) == tuple and len(x_sv) > 1:
        x_sv, others_sv = x_sv[0], x_sv[1:]
      else:
        x_sv = x_sv[0]
        others_sv = None
      y_sv = s_var(x_sv)
      y_sv, others_sv = s_var.call_hooks('postprocess_hooks', args=(y_sv, None))
      outputs.append(y_sv)
    
    output = self.output_head(outputs)
    output = output.unsqueeze(dim=0)"""
    return {'pred_logits': cls_pred, 'pred_boxes': bbox_pred}
  
  def set_backbone(self, backbone: dict)-> None:
    core = 'core.'
    backbone_pck = 'components.general_dl.backbone.' + backbone['type']
    dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), backbone_pck.replace('.', '/'))
    backbone_pck = core + backbone_pck
    backbone_cls = backbone['cls']
    cls = getattr(importlib.import_module(backbone_pck), backbone_cls)

    self.backbone_preinit_hooks = backbone['hooks']['preinit_hooks']

    args = make_arg(backbone['args'])
    kwargs = backbone['kwargs']
    kwargs['hooks'] = backbone['hooks']
    kwargs['device'] = self.device
    kwargs['base_dir'] = dir

    kwargs['preinit_return'] = self.call_hooks('backbone_preinit_hooks', args, kwargs)
    self.backbone = cls(*args, **kwargs).to(self.device, dtype=torch.float32)

  def set_detector(self, detector: dict) -> None:
    core = 'core.'
    detector_pck = 'components.general_dl.detectors.' + detector['type']
    dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), detector_pck.replace('.', '/'))
    detector_pck = core + detector_pck
    detector_cls = detector['cls']
    cls = getattr(importlib.import_module(detector_pck), detector_cls)

    self.detector_preinit_hooks = detector['hooks']['preinit_hooks']

    args = make_arg(detector['args'])
    kwargs = detector['kwargs']
    kwargs['hooks'] = detector['hooks']
    kwargs['device'] = self.device
    kwargs['base_dir'] = dir

    kwargs['preinit_return'] = self.call_hooks('detector_preinit_hooks', args, kwargs)
    self.detector = cls(*args, **kwargs).to(self.device, dtype=torch.float32)

  def set_semantic_variables(self, semantic_variables: dict):
    core = 'core.'
    for semantic_variable in semantic_variables:
      semantic_variable_pck = 'components.semantic_variables.' + \
                              semantic_variable['category'] + '.' + \
                              semantic_variable['type']
      dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), semantic_variable_pck.replace('.', '/'))
      semantic_variable_cls = semantic_variable['cls']
      semantic_variable_pck = core + semantic_variable_pck
      cls = getattr(importlib.import_module(semantic_variable_pck), semantic_variable_cls)

      self.semantic_variable_preinit_hooks = semantic_variable['hooks']['preinit_hooks']

      args = make_arg(semantic_variable['args'])
      kwargs = semantic_variable['kwargs']
      kwargs['hooks'] = semantic_variable['hooks']
      kwargs['device'] = self.device
      kwargs['base_dir'] = dir

      kwargs['preinit_return'] = self.call_hooks('semantic_variable_preinit_hooks', args, kwargs)
      kwargs['in_channels'] = self.get_input_channels()
      self.semantic_variables.append(cls(*args, **kwargs).to(self.device, dtype=torch.float32))

  def set_output_head(self, output_head: dict):
    core = 'core.'
    output_head_pck = 'components.output_heads.' + output_head['type']
    dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), output_head_pck.replace('.', '/'))

    output_head_pck = core + output_head_pck
    output_head_cls = output_head['cls']
    cls = getattr(importlib.import_module(output_head_pck), output_head_cls)

    self.output_head_preinit_hooks = output_head['hooks']['preinit_hooks']

    sv_outputs = output_head['detector_outputs']
    for sv in self.semantic_variables:
      sv_outputs.append(sv.get_outputs())

    args = make_arg(output_head['args'])
    args.insert(0, sv_outputs)
    args.insert(1, [self.img_channels, self.img_height, self.img_width])
    kwargs = output_head['kwargs']
    kwargs['hooks'] = output_head['hooks']
    kwargs['device'] = self.device
    kwargs['base_dir'] = dir

    kwargs['preinit_return'] = self.call_hooks('output_head_preinit_hooks', args, kwargs)
    self.output_head = cls(*args, **kwargs).to(self.device, dtype=torch.float32)
  
  def get_input_channels(self) -> Tuple[int]:
    return self.img_channels, self.img_channels, self.backbone.get_channels(), self.detector.get_channels()