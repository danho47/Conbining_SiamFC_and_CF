from __future__ import absolute_import

import os
import glob
import numpy as np
from tracker import TrackerModel
import shutil
import gc
import cv2 
import sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--tracker', default='SiamFC+CF', type=str,required=True)
parser.add_argument('--data', type=str, metavar='PATH')
args = parser.parse_args()
if __name__ == '__main__':
    
    #seq_dir = os.path.expanduser('/home/fujita/Desktop/honda/data/OTB/Couple/')
    seq_dir = args.data
    img_files = sorted(glob.glob(seq_dir + '/*.jpg'))
    #anno = np.loadtxt(seq_dir + 'groundtruth_rect.txt')
    tracker_name = args.tracker

    net_path = './pretrained/alexnet_e47.pth'
    tracker = TrackerModel(net_path=net_path)
    tracker.track(img_files, 'demo', tracker = tracker_name, visualize=True)



