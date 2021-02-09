
import os
import json
import random
from os.path import join

sample_random = random.Random()
class SingleData(object):
    
    def __init__(self,data_name, start):
        self.data_name = data_name
        self.start = start
        

        
        if data_name == 'YTB': 
            self.root = '/home/fujita/Desktop/honda/data/y2b/crop271'
            self.ano = '/home/fujita/Desktop/honda/data/y2b/train.json'
            self.frame_range = 3
            self.num_use = 200000
        elif data_name == 'VID':
            self.root = '/home/fujita/Desktop/honda/data/SiamDW_data/VID/crop255'
            self.ano = '/home/fujita/Desktop/honda/data/SiamDW_data/VID/train.json'
            self.frame_range = 100
            self.num_use = 200000
        elif data_name == 'DET':
            self.root = '/home/fujita/Desktop/honda/data/det/det/crop511'
            self.ano = '/home/fujita/Desktop/honda/data/det/det/train.json'
            self.frame_range = 100
            self.num_use = 100000
        
        with open(self.ano) as fin:
            self.labels = json.load(fin)
            self._clean()
            self.num=len(self.labels)
        
        self._shuffle()
    
    def _clean(self):
        
        to_del=[]
        for video in self.labels:
            for track in self.labels[video]:
                frames = self.labels[video][track]
                frames = list(map(int, frames.keys()))
                frames.sort()
                self.labels[video][track]['frames'] = frames
                
                if len(frames) <= 0:
                    print("warning {}/{} has no frames.".format(video, track))
                    to_del.append((video, track))
        
        for video, track in to_del:
            del self.labels[video][track]
        
        to_del = []
         
        if self.data_name == 'YTB':
            to_del.append('train/1/YyE0clBPamU')
            
        for video in self.labels:
            if len(self.labels[video]) <= 0:
                print("warning {} has no tracks".format(video))
                to_del.append(video)
            
        for video in to_del:
            del self.labels[video]
        
        self.videos = list(self.labels.keys())
        print('{} loaded.'.format(self.data_name))
    
    def _shuffle(self):
        """
        shuffel to get random pairs index (video)
        """
        lists = list(range(self.start, self.start + self.num))    
        m = 0
        pick = []
        while m < self.num_use:
            sample_random.shuffle(lists)
            pick += lists
            m += self.num
        
        self.pick = pick[:self.num_use]
        return self.pick
    
    def _get_image_anno(self, video, track, frame):
        
        frame = "{:06d}".format(frame)
        image_path = join(self.root, video, "{}.{}.x.jpg".format(frame, track))
        image_anno = self.labels[video][track][frame]
        
        return image_path,image_anno
    
    def _get_pairs(self,index):
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = random.choice(list(video.keys()))
        track_info = video[track]
        try:
            frames = track_info['frames']
        except:
            frames = list(track_info.keys())
            
        
        template_frame= random.randint(0, len(frames)-1)
        
        left = max(template_frame - self.frame_range, 0)
        right = min(template_frame + self.frame_range, len(frames)-1) + 1
        search_range = frames[left:right]
        template_frame = frames[template_frame]
        search_frame = random.choice(search_range)
        
        return self._get_image_anno(video_name, track, template_frame), \
               self._get_image_anno(video_name, track, search_frame)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        