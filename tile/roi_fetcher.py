import numpy as np
import pybgs as bgs
import cv2
from typing import List
try:
    from rect import *
    from filter import FilterFunc
    from inflater import InflateFunc
except:
    from .rect import *
    from .filter import FilterFunc
    from .inflater import InflateFunc
from codec import Frame

valid_algorithm_names = [
    'FrameDifference', 'ViBe', 'StaticFrameDifference', 'TwoPoints'
]

def get_algorithm_by_name(alg_name: str):
    if alg_name == 'FrameDifference':
        return bgs.FrameDifference()
    elif alg_name == 'StaticFrameDifference':
        return bgs.StaticFrameDifference()
    elif alg_name == 'ViBe':
        return bgs.ViBe()
    elif alg_name == 'TwoPoints':
        return bgs.TwoPoints()
    else:
        return None

class ROIFetcher:
    def __init__(self, 
        alg_name: str=None,
        scale_factor: float=0.25,               # scaling (usually downsample) the frame to speedup 
        blur_kernel: int=5,
        filter_list: List[FilterFunc]=None,     # functions to filter unvalid bbox
        inflater_func: InflateFunc=None
    ) -> None:
        '''
        A fast roi fetcher with background split methods
        '''
        self.alg = None
        if alg_name != None:
            self.use(alg_name)
        #self.foreground_list = []
        self.scale_factor = scale_factor
        self.blur_kernel = blur_kernel
        self.filter_list = filter_list
        self.inflater_func = inflater_func

    def use(self, alg_name:str):
        assert alg_name in valid_algorithm_names, f'{alg_name} is not supported by pybgs'
        self.alg = get_algorithm_by_name(alg_name)

    def pre_input(self, frame: Frame):
        '''
        this function is used to pre-set a frame in the bgs
        you can use it when using StaticFrameDifference
        '''
        if self.alg is None:
            self.alg = bgs.StaticFrameDifference()
            print('using static frame difference')
        frame = cv2.resize(frame, None, fx=self.scale_factor, fy=self.scale_factor)
        frame = cv2.GaussianBlur(frame, (self.blur_kernel, self.blur_kernel), 0)
        self.alg.apply(frame)

    def fetch(self, frame: Frame) -> List[Rect]: # (h, w, c)
        if self.alg is None:
            self.alg = bgs.FrameDifference()
            print('using default algorithm')
        frame = cv2.resize(frame, None, fx=self.scale_factor, fy=self.scale_factor)
        foreground = self.alg.apply(frame)
        foreground = self.post_process(foreground)
        return self.get_roi(foreground)
        
    def post_process(self, foreground:np.array) -> np.array:
        kernel = np.ones((3, 3), np.uint8)
        foreground = cv2.dilate(foreground, kernel)
        return foreground
    
    def get_roi(self, foreground:np.array) -> List[Rect]:
        contours, _ = cv2.findContours(foreground, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        roi_list = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            roi = Rect(x1=x, y1=y, w=w, h=h, scale_factor=1./self.scale_factor)
            prune = False
            if self.filter_list != None:
                for filter in self.filter_list:
                    if filter(roi):
                        prune = True
                        break
            if not prune:
                if self.inflater_func is not None:
                    roi = self.inflater_func(roi)
                roi_list.append(roi)
        return roi_list

    '''def combine(self) -> np.array:
        mask = self.masks[-1]
        for i in range(len(self.masks[i])-1):
            mask += self.masks[i]
        self.masks = []
        return mask'''

if __name__ == '__main__':
    skip_area = [Rect(40, 15, w=450, h=20), Rect(120, 475, w=250, h=50)]
    from filter import create_area_filter, create_size_filter
    area_filter = create_area_filter(skip_area)
    size_filter = create_size_filter([10, 10])
    f = ROIFetcher('ViBe', filter_list=[area_filter, size_filter])
    import sys
    sys.path.append('.')
    from codec import CV2Decoder, PyAVEncoder, PyAVEncoderConfig
    dec = CV2Decoder('./data/warsaw.mp4')
    frames = []
    cnt = 0
    while cnt < 150:
        frame = dec.get_frame()
        if frame is None:
            break
        frames.append(frame)
        cnt += 1
    ann_frame = []
    for frame in frames:
        roi_list = f.fetch(frame)
        for roi in roi_list:
            frame = cv2.rectangle(frame, float2int(roi.lt()), float2int(roi.rb()), (255,0,0), 2)
        for area in skip_area:
            frame = cv2.rectangle(frame, float2int(area.lt()), float2int(area.rb()), (0,0,255), 2)
        ann_frame.append(frame)
    import os
    if os.path.exists('./out.mp4'):
        os.remove('./out.mp4')
    cfg = PyAVEncoderConfig()
    #cfg.codec = 'h264'
    enc = PyAVEncoder('./out.mp4', cfg)
    enc.start()
    enc.encode(ann_frame)
    enc.finish()
    enc.join()