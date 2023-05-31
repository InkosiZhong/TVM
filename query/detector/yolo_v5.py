import torch, torchvision
from torch.nn.functional import interpolate
import numpy as np
import json
import time
import sys, os
sys.path.append('.')
from typing import List, Tuple
from tile import Rect
try:
    from base_detector import BaseDetector, ObjectInfo
except:
    from .base_detector import BaseDetector, ObjectInfo

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results
    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output

class YoloV5(BaseDetector):
    def __init__(self, 
                fx: float=1., 
                fy: float=1., 
                tx: float=0, 
                ty: float=0, 
                scale_factor: float=0.5,
                model_capacity: str='yolov5s', 
                map_func=lambda x: x
        ) -> None:
        self.fx = fx
        self.fy = fy
        self.tx = tx
        self.ty = ty
        self.model_capacity = model_capacity
        #self.model = torch.hub.load('ultralytics/yolov5', model_capacity, pretrained=True).eval()
        self.model = torch.hub.load('./third_party/ultralytics_yolov5_master', 'custom', source='local', path=f'{model_capacity}.pt').eval()
        self.names = self.model.names
        super().__init__(scale_factor, map_func)

    def export(self, name, model):
        traced_graph = torch.jit.trace(model, torch.zeros(1, 3, 224, 224))
        traced_graph.save(os.path.join(self.cache_path, name))

    def load(self, name) -> bool:
        cache = os.path.join(self.cache_path, name)
        if os.path.exists(cache):
            self.model = torch.jit.load(cache)
            print(f'loaded cache from {cache}')
            return True
        return False
    
    def get_info(self) -> dict:
        try:
            return json.load(open('dnn.json', 'r'))[self.model_capacity]
        except:
            print(f'warning: cannot load dnn info for {self.model_capacity}')
            return super().get_info()

    def cuda(self):
        self.model = self.model.cuda()
        return self

    def eval(self):
        self.model = self.model.eval()
        return self

    def preprocess(self, x: torch.Tensor):
        h, w = x.shape[2:]
        new_h = max(int(h * self.scale_factor // 64 * 64), 64)
        new_w = max(int(w * self.scale_factor // 64 * 64), 64)
        x = interpolate(x, (new_h, new_w))
        return x, w / new_w, h / new_h

    def interface(self, x: torch.Tensor, offset: Tuple[float, float]=(0, 0)) -> List[List[ObjectInfo]]:
        x, fx1, fy1 = self.preprocess(x)
        with torch.no_grad():
            all_objects = self.model(x)
        all_objects = non_max_suppression(all_objects, max_det=300)
        all_labels = []
        for objects in all_objects:
            labels = []
            for obj in objects:
                x1, y1, x2, y2, conf, class_idx = obj.cpu().numpy()
                bbox = Rect(x1 * self.fx * fx1 + self.tx, y1 * self.fy * fy1 + self.ty, 
                    x2 * self.fx * fx1 + self.tx, y2 * self.fy * fy1 + self.ty)
                labels.append(ObjectInfo(self.names[class_idx], conf, bbox, offset))
            all_labels.append(labels)
        return all_labels

if __name__ == '__main__':
    import time
    model = YoloV5(scale_factor=1, model_capacity='yolov5m').eval().cuda()
    SIZE = 100
    x = torch.zeros([1,3,SIZE,SIZE]).cuda()
    t = time.time()
    with torch.no_grad():
        _ = model(x)
    print(time.time() - t)