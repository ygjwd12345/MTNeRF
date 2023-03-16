import os, sys

sys.path.append('..')
import numpy as np
import imageio
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
import cv2
import pdb
from utils.io_utils import *
from options import config_parser
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def train(args):
    image_list = load_img_list(args.datadir, load_test=True)

    ### add depth loss
    gt_depths, _ = load_gt_depths(image_list, args.datadir,480, 640)
    i=0
    for gt in gt_depths:
        # print(np.max(gt))
        # print(np.min(gt))
        frame_id = image_list[i].split('.')[0]
        disp_visual = visualize_depth(gt,depth_min=np.min(gt), depth_max=0.6*np.max(gt))
        filename = os.path.join('./gt', '{}_gt_depth.png'.format(frame_id))
        cv2.imwrite(filename, disp_visual)
        i+=1





if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    parser = config_parser()
    args = parser.parse_args()
    train(args)