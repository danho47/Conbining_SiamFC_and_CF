import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau

from siamfc.ytb import YTB_BB
from got10k.datasets import *
from siamfc.datasets import Pair
import numpy as np
import cv2
from PIL import Image
root = '/home/fujita/Desktop/honda/data/y2b/crop271'
ano = '/home/fujita/Desktop/honda/data/y2b/train.json'
train_set = YTB_BB(root, ano)
print('type : ', type(train_set))
print('len : ',len(train_set))
load = DataLoader(train_set, batch_size = 16, num_workers=16,  pin_memory=True, sampler=None)
print(load)
print('type :',type(load))
print('len : ',len(load))
#template_image = cv2.imread('/home/fujita/Desktop/honda/data/y2b/train/9/nOlORwAI0bs/000406.00.x.jpg')
#print(type(template_image))
#print(template_image)
img = Image.open('/home/fujita/Desktop/honda/data/y2b/crop271/train/9/nOlORwAI0bs/000406.00.x.jpg')
img.show()

root_dir = '/home/fujita/Desktop/honda/SiamMask/data/vid/ILSVRC2015'
seqs = ImageNetVID(root_dir, subset=('train'))
print('type : ', type(seqs))
print('len_Vid : ',len(seqs))
print('vid :',seqs)
indices = np.random.permutation(len(seqs))
index = 100 
print('seqs[index] :', seqs[index])

#dataset = Pair(seqs=seqs,transforms=transforms)
#print('type ',type(dataset))
#print('len_vid :',len(dataset))
#print(dataset)