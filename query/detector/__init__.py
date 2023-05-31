from .base_detector import ObjectInfo
from .yolo_v5 import YoloV5
from .mask_rcnn import MaskRCNN

__all__ = [
    'ObjectInfo',
    'YoloV5',
    'MaskRCNN'
]