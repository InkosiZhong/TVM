import torch, torchvision
import sys
sys.path.append('.')
from typing import List, Tuple
from tile import Rect
try:
    from base_detector import BaseDetector, ObjectInfo
except:
    from .base_detector import BaseDetector, ObjectInfo

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

class MaskRCNN(BaseDetector):
    def __init__(self, fx:float=1., fy:float=1., tx: float=0, ty: float=0, map_func=lambda x: x) -> None:
        super().__init__(1, map_func)
        self.fx = fx
        self.fy = fy
        self.tx = tx
        self.ty = ty
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=True).eval()
        self.names = COCO_INSTANCE_CATEGORY_NAMES

    def cuda(self):
        self.model = self.model.cuda()
        return self

    def eval(self):
        self.model = self.model.eval()
        return self

    def interface(self, x: torch.Tensor, offset: Tuple[float, float]=(0, 0)) -> List[List[ObjectInfo]]:
        with torch.no_grad():
            all_objects = self.model(x)
        all_labels = []
        for objects in all_objects:
            boxes = objects['boxes'].detach().cpu().numpy()
            confidences = objects['scores'].detach().cpu().numpy()
            object_ids = objects['labels'].detach().cpu().numpy()
            labels = []
            for class_idx, box, conf in zip(object_ids, boxes, confidences):
                object_name = COCO_INSTANCE_CATEGORY_NAMES[class_idx]
                bbox = Rect(box[0] * self.fx + self.tx, box[1] * self.fy + self.ty, box[2] * self.fx + self.tx, box[3] * self.fy + self.ty)
                labels.append(ObjectInfo(object_name, conf, bbox, offset))
            all_labels.append(labels)
        return all_labels