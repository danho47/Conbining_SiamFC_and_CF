from __future__ import absolute_import

import os
from got10k.datasets import *

from tracker import TrackerModel
from tracker.ytb import YTB_BB

if __name__ == '__main__':
    root_dir = os.path.expanduser('/home/fujita/Desktop/honda/data/GOT-10k')
#    root_dir = os.path.expanduser('/home/honda/data/ILSVRC')
    #seqs = GOT10k(root_dir, subset='train', return_meta=True)
#    seqs = ImageNetVID(root_dir, subset=('train', 'val'))

    train_set = YTB_BB()
    tracker = TrackerSiamFC()
    tracker.train_over(train_set)
