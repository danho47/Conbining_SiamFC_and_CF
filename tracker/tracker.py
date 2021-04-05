from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
#import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import cv2
import sys
import os
from collections import namedtuple
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from got10k.trackers import Tracker
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from . import ops
from .heads import SiamFC
from .losses import BalancedLoss
from .datasets import Pair
from .transforms import SiamFCTransforms
from .backbones import AlexNetV1


from CFtrackers.cftracker.staple import Staple
#from pyCFTrackers.cftracker.bacf import BACF
#from pyCFTrackers.cftracker.csrdcf import CSRDCF
#from pyCFTrackers.cftracker.kcf import KCF
#from pyCFTrackers.cftracker.mccth_staple import MCCTHStaple
#from pyCFTrackers.cftracker.ldes import LDES
__all__ = ['TrackerModel']


class Net(nn.Module):

    def __init__(self, backbone, head):
        super(Net, self).__init__()
        self.backbone = backbone
        self.head = head
    
    def forward(self, z, x):
        z = self.backbone(z)
        x = self.backbone(x)
        return self.head(z, x)


class TrackerModel(Tracker):

    def __init__(self, net_path=None, **kwargs):
        super(TrackerModel, self).__init__('SiamFC', True)
        self.cfg = self.parse_args(**kwargs)

        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda')
        # setup model
        self.net = Net(
            backbone=AlexNetV1(),
            head=SiamFC(self.cfg.out_scale))
        ops.init_weights(self.net)
        
        # load checkpoint if provided
        if net_path is not None:
            self.net = self.net.to(self.device)
            self.net.backbone = nn.DataParallel(self.net.backbone)
            self.net.load_state_dict(torch.load(
                net_path, map_location=lambda storage, loc: storage))

        #self.net = self.net.to(self.device)
        #self.net.backbone = nn.DataParallel(self.net.backbone)

        # setup criterion
        self.criterion = BalancedLoss()

        # setup optimizer
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=self.cfg.initial_lr,
            weight_decay=self.cfg.weight_decay,
            momentum=self.cfg.momentum)
        
        # setup lr scheduler
        gamma = np.power(
            self.cfg.ultimate_lr / self.cfg.initial_lr,
            1.0 / self.cfg.epoch_num)
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma)

    def parse_args(self, **kwargs):
        # default parameters
        cfg = {
            # basic parameters
            'out_scale': 0.001,
            'exemplar_sz': 127,
            'instance_sz': 255,
            'context': 0.5,
            # inference parameters
            'scale_num': 3,
            'scale_step': 1.0375,
            'scale_lr': 0.59,
            'scale_penalty': 0.9745,
            'window_influence': 0.176,
            'response_sz': 17,
            'response_up': 16,
            'total_stride': 8,
            # train parameters
            'epoch_num': 50,
            'batch_size': 128,
            'num_workers': 16,
            'initial_lr': 1e-2,
            'ultimate_lr': 1e-5,
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'r_pos': 16,
            'r_neg': 0
            }
        for key, val in kwargs.items():
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('Config', cfg.keys())(**cfg)
    
    @torch.no_grad()
    def init(self, img, box):
        # set to evaluation mode
        self.net.eval()
        #img = np.asarray(img)
        self.cf = Staple()
        #bbox = tuple(box)
        #print(bbox)
        self.cf.init(img,box)       
        
        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]

        # create hanning window
        self.upscale_sz = self.cfg.response_up * self.cfg.response_sz
        self.hann_window = np.outer(
            np.hanning(self.upscale_sz),
            np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()

        # search scale factors
        self.scale_factors = self.cfg.scale_step ** np.linspace(
            -(self.cfg.scale_num // 2),
            self.cfg.scale_num // 2, self.cfg.scale_num)

        # exemplar and search sizes
        context = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
            self.cfg.instance_sz / self.cfg.exemplar_sz
        
        # exemplar image
        self.avg_color = np.mean(img, axis=(0, 1))
        z = ops.crop_and_resize(
            img, self.center, self.z_sz,
            out_size=self.cfg.exemplar_sz,
            border_value=self.avg_color)

        # exemplar features
        z = torch.from_numpy(z).to(
            self.device).permute(2, 0, 1).unsqueeze(0).float()
        self.kernel = self.net.backbone(z)
        
    
    
    @torch.no_grad()
    def update(self, img, frame, tracker):
        # set to evaluation mode
        self.net.eval() 
             
        # search images
        x = [ops.crop_and_resize(
            img, self.center, self.x_sz * f,
            out_size=self.cfg.instance_sz,
            border_value=self.avg_color) for f in self.scale_factors]
        x = np.stack(x, axis=0)
        
        x = torch.from_numpy(x).to(
            self.device).permute(0, 3, 1, 2).float()
        # responses
        x = self.net.backbone(x)
        responses = self.net.head(self.kernel, x)
        responses = responses.squeeze(1).cpu().numpy()

        # upsample responses and penalize scale changes
        responses = np.stack([cv2.resize(
            u, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC)
            for u in responses])
        responses[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty
        responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty


        # peak scale
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        # peak location
        response = responses[scale_id]
        response -= response.min()
        response /= response.sum() + 1e-16
        response = (1 - self.cfg.window_influence) * response + \
            self.cfg.window_influence * self.hann_window
        loc = np.unravel_index(response.argmax(), response.shape)
        apce = self.APCE(response)
        psr = self.PSR(response)
        score = 0.9*apce + 0.1*psr
        #print('SiamFC :',apce)

        #Corrrelation Filter
        if tracker == 'SiamFC+CF':
            region = self.cf.update(img)
            apce_cf = self.APCE(self.cf.score)
            psr_cf = self.PSR(self.cf.score)
            score_cf = 0.9*apce_cf + 0.1*psr_cf
            #print('CF :',apce_cfps)


        # locate target center
        disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2
        disp_in_instance = disp_in_response * \
            self.cfg.total_stride / self.cfg.response_up
        disp_in_image = disp_in_instance * self.x_sz * \
            self.scale_factors[scale_id] / self.cfg.instance_sz
        self.center += disp_in_image

        # update target size
        scale =  (1 - self.cfg.scale_lr) * 1.0 + \
            self.cfg.scale_lr * self.scale_factors[scale_id]
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale

        # return 1-indexed and left-top based bounding box
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])
       
        # Judging Confidence
        if tracker == 'SiamFC+CF':
            if score<5.0:
                if score_cf > score:
                    box = region
                    self.center = np.array([box[1]+(box[3]/2),
                                            box[0]+(box[2]/2)])
        else:
            box = box
            
      
        return box
    
    def track(self, img_files, box, tracker='SiamFC+CF', visualize=False):
        frame_num = len(img_files)
        img = ops.read_image(img_files[0])

        self.boxes = np.zeros((frame_num, 4))
        if box == 'demo':
            #cv2.namedWindow("box", cv2.WND_PROP_FULLSCREEN)
            init_box = cv2.selectROI("box", img, showCrosshair=False, fromCenter=False)
            self.boxes[0] = init_box
            box = init_box
        else:
            self.boxes[0] = box
        times = np.zeros(frame_num)
        
        # parse batch data
        for f, img_file in enumerate(img_files):
            #img = cv2.imread(img_file)
            img = ops.read_image(img_file)
            begin = time.time()
            if f == 0:
                self.init(img, box)
            else:
                if tracker == 'SiamFC+CF' or tracker == 'SiamFC':
                    self.boxes[f, :] = self.update(img, f, tracker)
                    
                elif tracker =='Staple':
                    self.boxes[f, :] = self.cf.update(img)

            times[f] = time.time() - begin

            if visualize:
                ops.show_image(img, self.boxes[f, :])
        return self.boxes, times
        
    
    def train_step(self, batch, backward=True):
        # set network mode
        self.net.train(backward)

        z = batch[0].to(self.device, non_blocking=self.cuda)
        x = batch[1].to(self.device, non_blocking=self.cuda)
#        print('x: ',len(x[1]))
#        print('z: ',len(z[1]))

        with torch.set_grad_enabled(backward):
            # inference
            responses = self.net(z, x)
#            print('response: ',len(responsess))

            # calculate loss
            labels = self._create_labels(responses.size())
            loss = self.criterion(responses, labels)
            
            if backward:
                # back propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        return loss.item()

    @torch.enable_grad()
    def train_over(self, data, val_seqs=None,
                   save_dir='pretrained/DET_YTB/'):
        # set to train mode
        self.net.train()
        writer = SummaryWriter()

        # create save_dir folder
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # setup dataset
        transforms = SiamFCTransforms(
            exemplar_sz=self.cfg.exemplar_sz,
            instance_sz=self.cfg.instance_sz,
            context=self.cfg.context)
#        dataset = Pair(
#            seqs=seqs,
#            transforms=transforms)
        
        # setup dataloader
        dataloader = DataLoader(
            data,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cuda,
            drop_last=True)
        
        # loop over epochs
        for epoch in range(self.cfg.epoch_num):
            # update lr at each epoch
            self.lr_scheduler.step(epoch=epoch)

            # loop over dataloader
            for it, batch in enumerate(dataloader):
                loss = self.train_step(batch, backward=True)
                print('Epoch: {} [{}/{}] Loss: {:.5f}'.format(
                    epoch + 1, it + 1, len(dataloader), loss))
                writer.add_scalar("Total_Loss/train",loss,epoch)
                writer.flush() 
                sys.stdout.flush()
                

            
	                 
            # save checkpoint
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            net_path = os.path.join(
                save_dir, 'alexnet_e%d.pth' % (epoch + 1))
            torch.save(self.net.state_dict(), net_path)
        
    def cv2pil(self, image_cv):
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_cv)
        image_pil = image_pil.convert('RGB')
    
        return image_pil
    
    def _create_labels(self, size):
        # skip if same sized labels already created
        if hasattr(self, 'labels') and self.labels.size() == size:
            return self.labels

        def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)  # block distance
            labels = np.where(dist <= r_pos,
                              np.ones_like(x),
                              np.where(dist < r_neg,
                                       np.ones_like(x) * 0.5,
                                       np.zeros_like(x)))
            return labels

        # distances along x- and y-axis
        n, c, h, w = size
        x = np.arange(w) - (w - 1) / 2
        y = np.arange(h) - (h - 1) / 2
        x, y = np.meshgrid(x, y)

        # create logistic labels
        r_pos = self.cfg.r_pos / self.cfg.total_stride
        r_neg = self.cfg.r_neg / self.cfg.total_stride
        labels = logistic_labels(x, y, r_pos, r_neg)

        # repeat to size
        labels = labels.reshape((1, 1, h, w))
        labels = np.tile(labels, (n, c, 1, 1))

        # convert to tensors
        self.labels = torch.from_numpy(labels).to(self.device).float()
        
        return self.labels
    
    def IOU(self, a, b):
        # get area of a
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        # get area of b
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        
        # get left top x of IoU
        iou_x1 = np.maximum(a[0], b[0])
        # get left top y of IoU
        iou_y1 = np.maximum(a[1], b[1])
        # get right bottom of IoU
        iou_x2 = np.minimum(a[2], b[2])
        # get right bottom of IoU
        iou_y2 = np.minimum(a[3], b[3])
    
        # get width of IoU
        iou_w = iou_x2 - iou_x1
        # get height of IoU
        iou_h = iou_y2 - iou_y1
    
        # no overlap
        if iou_w < 0 or iou_h < 0:
            return 0.0
        
        # get area of IoU
        area_iou = iou_w * iou_h
        # get overlap ratio between IoU and all area
        iou = area_iou / (area_a + area_b - area_iou)
        union = (area_a + area_b - area_iou)
    
        return iou
    def APCE(self,response_map):
        Fmax=np.max(response_map)
        Fmin=np.min(response_map)
        apce=(Fmax-Fmin)**2/(np.mean((response_map-Fmin)**2))
        return apce
    
    def PSR(self,response):
        response_map=response.copy()
        max_loc=np.unravel_index(np.argmax(response_map, axis=None),response_map.shape)
        y,x=max_loc
        F_max = np.max(response_map)
        response_map[y-5:y+6,x-5:x+6]=0.
        mean=np.mean(response_map[response_map>0])
        std=np.std(response_map[response_map>0])
        psr=(F_max-mean)/std
        return psr
