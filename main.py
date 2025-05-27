from PIL import Image
import os, copy, argparse

import torch, torch.nn as nn, torch.nn.functional as F

from builder import builder
from trainer import Trainer
from core import base_model
from core.util import DotDict
from torch.utils.data import DataLoader

def args_to_config(args, config):
    config['trainer']['resume'] = args.r
    config['trainer']['start_epoch'] = args.se
    config['trainer']['epochs'] = args.ne
    config['trainer']['eval'] = args.e
    return config

def start(args = None, config: dict = None):
    dir = os.path.dirname(os.path.realpath(__file__))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = config if config is not None else builder.Builder.load_config(f'test_config')
    config = fix_bs(args, config)
    config = args_to_config(args, config)
    build = builder.Builder(config)
    model = build.get_model(device)
    datasets =  build.get_datasets(device)
    criterion = # Define some criterion
    
    trainer = Trainer(build.config['trainer'], device)
    trainer.run(model, datasets["train"], datasets["eval"], criterion)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-bs', type=int, default=1,
                        help='batch size')
    parser.add_argument('-r', type=str, default=None,
                        help='resume from file')
    parser.add_argument('-se', type=int, default=0,
                        help='starting epoch')
    parser.add_argument('-ne', type=int, default=30,
                        help='number of epochs')
    parser.add_argument('-e', action='store_true',
                        help='if eval')

    args = parser.parse_args()
    start(args=args)