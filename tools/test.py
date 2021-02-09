from __future__ import absolute_import

import os
from got10k.experiments import *
import glob

from tracker import TrackerModel


if __name__ == '__main__':
    net_path = '../pretrained/alexnet_e47.pth'
    tracker = TrackerModel(net_path=net_path)
    root_dir = os.path.expanduser('/home/fujita/Desktop/honda/data/OTB')
    e = ExperimentOTB(root_dir, version=2015)

    e.run(tracker,visualize=True)
    e.report([tracker.name])
    
