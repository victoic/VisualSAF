def make_arg(args):
    return_args = []
    for k, v in args.items():
      value = DotDict(v['value']) if type(v['value']) is dict else v['value']
      return_args.insert(v['index'], value)
    return return_args

class DotDict(dict):
  """dot.notation access to dictionary attributes"""
  def __getattr__(*args):
    val = dict.get(*args)
    return DotDict(val) if type(val) is dict else val
  setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__