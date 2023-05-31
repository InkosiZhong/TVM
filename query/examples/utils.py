from scipy.spatial import distance
from tile import Rect, intersect_rect
from typing import List
from functools import partial
import torch
from torch.nn.functional import interpolate
from tile import intersect_rect

def count_is_close_helper(label1, label2):
    return len(label1) == len(label2)

def position_is_close_helper(label1, label2):
    if len(label1) != len(label2):
        return False
    counter = 0
    for obj1 in label1:
        xavg1 = (obj1.xmin + obj1.xmax) / 2.0
        yavg1 = (obj1.ymin + obj1.ymax) / 2.0
        coord1 = [xavg1, yavg1]
        expected_counter = counter + 1
        for obj2 in label2:
            xavg2 = (obj2.xmin + obj2.xmax) / 2.0
            yavg2 = (obj2.ymin + obj2.ymax) / 2.0
            coord2 = [xavg2, yavg2]
            if distance.euclidean(coord1, coord2) < 100:
                counter += 1
                break
        if expected_counter != counter:
            break
    return len(label1) == counter

def template_score_func(target_dnn_output, valid_area: Rect, ban_area: List[Rect], classes: List[str]):
    cnt = 0
    for output in target_dnn_output:
        if output.object_name not in classes:
            continue
        inter = intersect_rect(output.abs_bbox(), valid_area)
        if inter is not None and inter.area() >= 0.7 * output.bbox.area():
            ok = True
            for rect in ban_area:
                inter = intersect_rect(output.abs_bbox(), rect)
                if inter is not None and inter.area() >= 0.3 * output.bbox.area():
                    ok = False
                    break
            if ok:
                cnt += 1
    return cnt

def create_score_func(valid_area: Rect, ban_area: List[Rect], classes: List[str]):
    return partial(template_score_func, valid_area=valid_area, ban_area=ban_area, classes=classes)

def template_embedding_dnn_transform_fn(frame, tile, valid_area, mask, target_size):
    rect = intersect_rect(tile, valid_area)
    xmin, ymin = int(rect.x1 - tile.x1), int(rect.y1 - tile.y1)
    xmax, ymax = int(xmin + rect.w), int(ymin + rect.h)
    frame = frame[:, ymin:ymax, xmin:xmax]
    mask_ = mask[int(rect.y1):int(rect.y2), int(rect.x1):int(rect.x2)]
    frame *= mask_
    t, l = 0, 0
    b, r = mask_.shape
    step = 10
    while torch.all(mask_[t+step] == 0):
        t += step
    while torch.all(mask_[b-step] == 0):
        b -= step
    while torch.all(mask_[t:b, l+step] == 0):
        l += step
    while torch.all(mask_[t:b, r-step] == 0):
        r -= step
    if b > t and r > l:
        frame = frame[:, t:b, l:r]
    frame = interpolate(frame.unsqueeze(0), target_size)
    return frame.view(3, *target_size)

def create_embedding_dnn_transform_fn(frame_shape, valid_area, ban_area, target_size=(224, 224), device=0):
    mask = torch.full(frame_shape, 1, device=device)
    for area in ban_area:
        mask[int(area.y1):int(area.y2),int(area.x1):int(area.x2)] = 0
    return partial(template_embedding_dnn_transform_fn, valid_area=valid_area, mask=mask, target_size=target_size)