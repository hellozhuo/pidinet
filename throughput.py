"""
(Testing FPS)
Pixel Difference Networks for Efficient Edge Detection (accepted as an ICCV 2021 oral)
See paper in https://arxiv.org/abs/2108.07009

Author: Zhuo Su, Wenzhe Liu
Date: Aug 22, 2020
"""

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import argparse
import os
import time
import models
from utils import *
from edge_dataloader import BSDS_VOCLoader, BSDS_Loader, Multicue_Loader, NYUD_Loader
from torch.utils.data import DataLoader

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

parser = argparse.ArgumentParser(description='PyTorch Diff Convolutional Networks (Train)')

parser.add_argument('--datadir', type=str, default='../data', 
        help='dir to the dataset')
parser.add_argument('--dataset', type=str, default='BSDS',
        help='data settings for BSDS, Multicue and NYUD datasets')

parser.add_argument('--model', type=str, default='baseline', 
        help='model to train the dataset')
parser.add_argument('--sa', action='store_true', 
        help='use attention in diffnet')
parser.add_argument('--dil', action='store_true', 
        help='use dilation in diffnet')
parser.add_argument('--config', type=str, default='nas-all', 
        help='model configurations, please refer to models/config.py for possible configurations')
parser.add_argument('--seed', type=int, default=None, 
        help='random seed (default: None)')
parser.add_argument('--gpu', type=str, default='', 
        help='gpus available')

parser.add_argument('--epochs', type=int, default=150, 
        help='number of total epochs to run')
parser.add_argument('-j', '--workers', type=int, default=4, 
        help='number of data loading workers')
parser.add_argument('--eta', type=float, default=0.3, 
        help='threshold to determine the ground truth')

args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

def main():

    global args

    ### Refine args
    if args.seed is None:
        args.seed = int(time.time())
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.use_cuda = torch.cuda.is_available()

    dataset_setting_choices = ['BSDS', 'NYUD-image', 'NYUD-hha', 'Multicue-boundary-1', 
                'Multicue-boundary-2', 'Multicue-boundary-3', 'Multicue-edge-1', 'Multicue-edge-2', 'Multicue-edge-3']
    if not isinstance(args.dataset, list): 
        assert args.dataset in dataset_setting_choices, 'unrecognized data setting %s, please choose from %s' % (str(args.dataset), str(dataset_setting_choices))
        args.dataset = list(args.dataset.strip().split('-')) 

    print(args)

    ### Create model
    model = getattr(models, args.model)(args)

    ### Transfer to cuda devices
    if args.use_cuda:
        if args.model == 'hed':
            model.weight_deconv2 = model.weight_deconv2.cuda()
            model.weight_deconv3 = model.weight_deconv3.cuda()
            model.weight_deconv4 = model.weight_deconv4.cuda()
            model.weight_deconv5 = model.weight_deconv5.cuda()
        model = torch.nn.DataParallel(model).cuda()
        print('cuda is used, with %d gpu devices' % torch.cuda.device_count())
    else:
        print('cuda is not used, the running might be slow')

    ### Load Data
    if 'BSDS' == args.dataset[0]:
        test_dataset = BSDS_VOCLoader(root=args.datadir, split="test", threshold=args.eta)
    elif 'Multicue' == args.dataset[0]:
        test_dataset = Multicue_Loader(root=args.datadir, split="test", threshold=args.eta, setting=args.dataset[1:])
    elif 'NYUD' == args.dataset[0]:
        test_dataset = NYUD_Loader(root=args.datadir, split="test", setting=args.dataset[1:])
    else:
        raise ValueError("unrecognized dataset setting")
    test_loader = DataLoader(
        test_dataset, batch_size=1, num_workers=args.workers, shuffle=False)

    test(test_loader, model, args)
    return


def test(test_loader, model, args):

    model.eval()

    end = time.perf_counter()
    torch.cuda.synchronize()
    for idx, (image, img_name) in enumerate(test_loader):

        with torch.no_grad():
            image = image.cuda() if args.use_cuda else image
            _, _, H, W = image.shape
            results = model(image)
    torch.cuda.synchronize()
    end = time.perf_counter() - end
    print('fps: %f' % (len(test_loader) / end))


if __name__ == '__main__':
    main()
    print('done')
