import numpy as np
from typing import List
import sys
sys.path.append('.')
sys.setrecursionlimit(100000)
from codec import Size, Segment
try:
    from rect import *
    from condition import ConditionFunc
    from filter import create_size_filter
    from utils import union_find, union, union_join
except:
    from .rect import *
    from .condition import ConditionFunc
    from .filter import create_size_filter
    from .utils import union_find, union, union_join

Layout = List[List[Tuple[Rect, bool]]] # 2D, bool specifies if this rect contains roi
Tile = Segment

class LayoutGenerator:
    def __init__(self, 
        frame_size: Size, 
        condition_list: List[ConditionFunc]=[
            intersect,
        ],
        prune_size: Size=(20, 20),
        min_size: Size=(0, 0),
    ) -> None:
        '''
        This class generates non-overlapping layout
        frame_size: (height, width) of the frame
        condition_list: functions to decide if 2 rect should merge
        filter_size: (height, width) of a ROI, any ROI smaller then it will be prune
        min_size: (height, width) of a tile, any tile smaller then it will be merged to its neighbor
        inflate_k: all roi will be scale according to this coefficient
        '''
        self.layout = None
        self.roi_list = []
        self.h = frame_size[0]
        self.w = frame_size[1]
        self.area = self.h * self.w
        self.condition_list = condition_list
        self.size_filter = create_size_filter(prune_size)
        self.min_h = min_size[0]
        self.min_w = min_size[1]

    def get_layout(self, roi_lists: List[List[Rect]]) -> Layout:
        self.roi_list = []
        frame_roi_list = []
        # first combine rois in each frame
        for roi_list in roi_lists:
            for condition in self.condition_list:
                roi_list = self.combine(roi_list, condition)
            self.roi_list += roi_list
            frame_roi_list.append(roi_list)
        # combine rois between frames
        self.roi_list = self.combine(self.roi_list, intersect)
        self.roi_list = [roi for roi in self.roi_list if not self.size_filter(roi)]
        self.gen_layout()
        return self.layout, frame_roi_list

    '''def combine(self, condition: ConditionFunc):
        parent = union(self.roi_list, condition)
        m = {}
        for i in range(len(self.roi_list)):
            p = find(i, parent)
            if p in m:
                m[p] = merge_rect(m[p], self.roi_list[i])
            else:
                m[p] = self.roi_list[i]
        self.roi_list = m.values()'''

    def combine(self, roi_list: List[Rect], condition: ConditionFunc):
        # larger rects have higher possibility to merge with others
        roi_list = sorted(roi_list, key=lambda r:r.area(), reverse=True)
        del_map = [False for _ in roi_list] # lazy delete
        no_relation = True
        for i in range(len(roi_list)):
            if del_map[i]:
                continue
            for j in range(i + 1, len(roi_list)):
                if del_map[j]:
                    continue
                if condition(roi_list[i], roi_list[j]):
                    del_map[j] = True  # this element will be deleted later
                    roi_list[i] = merge_rect(roi_list[i], roi_list[j])
                    no_relation = False
        if no_relation: # end of recurrence
            return roi_list
        # may be slow
        roi_list = [x for x, delete in zip(roi_list, del_map) if not delete]
        return self.combine(roi_list, condition)

    def gen_layout(self):
        xs, _ = self.gen_vertical_slice()
        ys, _ = self.gen_horizontal_slice()
        self.layout = []
        for i in range(len(ys)-1):
            line = []
            for j in range(len(xs)-1):
                r = Rect(xs[j], ys[i], xs[j+1], ys[i+1])
                #has_roi = y_has_roi[i] and x_has_roi[j] # this is wrong
                has_roi = False
                for roi in self.roi_list:
                    #if r.include(roi):
                    if r.contain(roi.ctr()):
                        has_roi = True
                        break
                line.append((r, has_roi))
            self.layout.append(line)

    def gen_vertical_slice(self):
        xs = [0]
        has_roi = []
        # vertical split first
        self.roi_list = sorted(self.roi_list, key=lambda r:r.x1)
        for roi in self.roi_list:
            x1 = self.norm(roi.x1)
            x2 = self.norm(roi.x2 + 1)
            if x1 > xs[-1]:
                xs.append(x1)
                xs.append(x2)
                has_roi.append(False)   # xs[-1] ~ x1
                has_roi.append(True)    # x1 ~ x2
            elif x1 == xs[-1]:
                if x2 > xs[-1]:
                    xs.append(x2)
                    has_roi.append(True)    # x1 ~ x2
            elif x2 > xs[-1]:       # these rois should in one tile
                if xs[-1] > 0:
                    xs[-1] = x2
                else:
                    xs.append(x2)
                    has_roi.append(True)
        if xs[-1] < self.w:
            xs.append(self.w)
            has_roi.append(False)    # xs[-1] ~ w
        xs, has_roi = self.min_size_filter(xs, has_roi, self.min_w)
        return xs, has_roi

    def gen_horizontal_slice(self):
        ys = [0]
        has_roi = []
        # vertical split first
        self.roi_list = sorted(self.roi_list, key=lambda r:r.y1)
        for roi in self.roi_list:
            y1 = self.norm(roi.y1)
            y2 = self.norm(roi.y2 + 1)
            if y1 > ys[-1]:
                ys.append(y1)
                ys.append(y2)
                has_roi.append(False)   # ys[-1] ~ y1
                has_roi.append(True)    # y1 ~ y2
            elif y1 == ys[-1]:
                if y2 > ys[-1]:
                    ys.append(y2)
                    has_roi.append(True)    # y1 ~ y2
            elif y2 > ys[-1]:       # these rois should in one tile
                if ys[-1] > 0:
                    ys[-1] = y2
                else:
                    ys.append(y2)
                    has_roi.append(True)
        if ys[-1] < self.h:
            ys.append(self.h)
            has_roi.append(False)    # ys[-1] ~ h
        ys, has_roi = self.min_size_filter(ys, has_roi, self.min_h)
        return ys, has_roi

    def norm(self, val, base=2):
        return int(val // base * base)

    def min_size_filter(self, arr: List[int], has_roi: List[bool], min_size: int):
        i = 0
        while i < len(arr) - 1:
            if arr[i+1] - arr[i] < min_size:
                if i == 0:              # first element
                    if has_roi[i] and not has_roi[i+1] and \
                        arr[i+2] - arr[i+1] >= 2 * min_size:
                        arr[i+1] = arr[i] + min_size    # adjust the tile containing roi
                    else:
                        arr.pop(i+1)                     # merge with the later one
                        has_roi.pop(i+1 if has_roi[i] else i)
                elif i == len(arr) - 2:  # last element
                    if has_roi[i] and not has_roi[i-1] and \
                        arr[i] - arr[i-1] >= 2 * min_size:
                        arr[i] = arr[i+1] - min_size    # adjust the tile containing roi
                    else:
                        arr.pop(i)                      # merge with the ahead one
                        has_roi.pop(i-1 if has_roi[i] else i)
                        i += 1
                else:
                    last_w = arr[i] - arr[i-1]
                    next_w = arr[i+2] - arr[i+1]
                    if last_w > next_w:
                        if has_roi[i] and not has_roi[i+1] and \
                            arr[i+2] - arr[i+1] >= 2 * min_size:
                            arr[i+1] = arr[i] + min_size
                        elif has_roi[i] and not has_roi[i-1] and \
                            arr[i] - arr[i-1] >= 2 * min_size:
                            arr[i] = arr[i+1] - min_size
                        else:
                            arr.pop(i+1)
                            has_roi.pop(i+1 if has_roi[i] else i)
                    else:
                        if has_roi[i] and not has_roi[i+1] and \
                            arr[i+2] - arr[i+1] >= 2 * min_size:
                            arr[i+1] = arr[i] + min_size
                        elif has_roi[i] and not has_roi[i-1] and \
                            arr[i] - arr[i-1] >= 2 * min_size:
                            arr[i] = arr[i+1] - min_size
                        else:
                            arr.pop(i)
                            has_roi.pop(i-1 if has_roi[i] else i)
                            i += 1
            else:
                i += 1
        return arr, has_roi

if __name__ == '__main__':
    import cv2
    def draw_layout(frame, layout):
        for line in layout:
            for l, has_roi in line:
                frame = cv2.rectangle(frame, float2int(l.lt()), float2int(l.rb()), (0,255,0) if has_roi else (0, 0, 255), 2 if has_roi else 1)
        return frame
    from filter import create_area_filter, create_size_filter
    # skip_area = [Rect(40, 15, w=450, h=20), Rect(120, 475, w=250, h=50)]
    skip_area = [Rect(0, 0, w=1920, h=400), Rect(0, 1060, w=1920, h=20)]
    from roi_fetcher import ROIFetcher
    f = ROIFetcher('ViBe', filter_list=[
        create_area_filter(skip_area),
        create_size_filter([5, 5])
    ])
    from codec import CV2Decoder, PyAVEncoder, PyAVEncoderConfig
    dec = CV2Decoder('../datasets/jackson-town-square/2017-12-14.mp4')
    from condition import create_edge_abs_close_condition
    t = LayoutGenerator((dec.height, dec.width), condition_list=[
        create_edge_abs_close_condition(0.02 * dec.height),
        intersect,
    ], min_size=(34, 130))
    gop_roi = []
    frames = []
    cnt = 0
    gop = 30
    ann_frames = []
    layout = []
    while cnt < 1500:
        frame = dec.get_frame()
        if frame is None:
            break
        if cnt > 0 and cnt % gop == 0:
            layout = t.get_layout(gop_roi)
            gop_roi = []
            for i in range(gop):
                ann_frames[cnt-1-i] = draw_layout(ann_frames[cnt-1-i], layout)
        frames.append(frame)
        roi = f.fetch(frame)
        for r in roi:
            frame = cv2.rectangle(frame, float2int(r.lt()), float2int(r.rb()), (255,0,0), 1)
            gop_roi.append(r)
        ann_frames.append(frame)
        cnt += 1
        
    import os
    if os.path.exists('./out.mp4'):
        os.remove('./out.mp4')
    cfg = PyAVEncoderConfig()
    #cfg.codec = 'h264'
    enc = PyAVEncoder('./out.mp4', cfg)
    enc.start()
    enc.encode(ann_frames)
    enc.finish()
    enc.join()