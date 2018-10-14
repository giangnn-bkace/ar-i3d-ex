import random
import os
import numpy as np
# To suppress the complaint of `image file is truncated`
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from PIL import Image
from PIL import ImageOps
from data_augment import transform_data



class Video_3D:
    def __init__(self, info_list, tag='rgb', img_format='frame{:06d}{}.jpg'):
        '''
            info_list: [name, path, total_frame, label]
            tag: 'rgb'(default) or 'flow'
            img_format: 'frame{:06d}{}.jpg'(default)
        '''
        #initialzie,to ensure the int is int
        self.name = info_list[0]
        self.path = info_list[1]
        if isinstance(info_list[2], int):
            self.total_frame_num = info_list[2]
        else:
            self.total_frame_num = int(info_list[2])
        if isinstance(info_list[3], int):
            self.label = info_list[3]
        else:
            self.label = int(info_list[3])
        self.tag = tag
        #img_format offer the standard name of pic
        self.img_format = img_format

    def get_frames(self, frame_num, side_length=224, is_numpy=True, data_augment=False):
        #assert frame_num <= self.total_frame_num
        frames = list()
        start = random.randint(1, max(self.total_frame_num-frame_num, 0)+1)
        #combine all frames
        for i in range(start, start+frame_num):
            frames.extend(self.load_img((i-1)%self.total_frame_num+1))
        frames = transform_data(frames, crop_size=side_length, random_crop=data_augment, random_flip=data_augment)

        
        
#?? what is the meaning of is_numpy
        if is_numpy:
            frames_np = []
            if self.tag == b'rgb':
                for i, img in enumerate(frames):
                    frames_np.append(np.asarray(img))
            elif self.tag == b'flow':
                for i in range(0, len(frames), 2):
                    #it is used to combine frame into 2 channels
                    tmp = np.stack([np.asarray(frames[i]), np.asarray(frames[i+1])], axis=2)
                    frames_np.append(tmp)
            return np.stack(frames_np)

        return  frames

    def get_frame_at(self, frame_num, start=1, sample=1, data_augment=False, random_start=False, side_length=224):
        '''
            return:
                frame_num * height * width * channel (rgb:3 , flow:2) 
        '''
        #assert frame_num <= self.total_frame_num
        start = start - 1
        #print('name: %s %d' % {self.name, start})
        if random_start:
            start = np.random.randint(max(self.total_frame_num-(frame_num-1)*sample, 1))
        frames = []
        for i in range(start, start+frame_num*sample, sample):
            frames.extend(self.load_img(i % self.total_frame_num + 1))
        frames = transform_data(frames, crop_size=side_length, random_crop=data_augment, random_flip=data_augment)
        frames_np = []
        if self.tag == 'rgb':
            for img in frames:
                frames_np.append(np.asarray(img))
        elif self.tag == 'flow':
            for i in range(0, len(frames), 2):
                tmp = np.stack([np.asarray(frames[i]), np.asarray(frames[i+1])], axis=2)
                frames_np.append(tmp)
        return frames_np
        
    def load_img(self, index):
        img_dir = self.path.decode('utf-8')
        if self.tag == b'rgb':
            img = Image.open(os.path.join(img_dir, self.img_format.format(index, ''))).convert('RGB')
            return [img]
        if self.tag == b'flow':
            # u_img = Image.open(os.path.join(img_dir, self.img_format.format(index, '_u'))).convert('L')
            # v_img = Image.open(os.path.join(img_dir, self.img_format.format(index, '_v'))).convert('L')
            u_img = Image.open(os.path.join(img_dir.format('u'), self.img_format.format(index, ''))).convert('L')
            v_img = Image.open(os.path.join(img_dir.format('v'), self.img_format.format(index, ''))).convert('L')
            return [u_img,v_img]
        return

    def __str__(self):
        return 'Video_3D:\nname: {:s}\nframes: {:d}\nlabel: {:d}\nPath: {:s}'.format(
            self.name, self.total_frame_num, self.label, self.path)
