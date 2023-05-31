import torch
from typing import List, Tuple
import sys, os, time
sys.path.append('.')
from tile import Rect, CostFunc, create_dnn_cost_func

class ObjectInfo:
    def __init__(self, object_name: str, conf: float, bbox: Rect, offset: Tuple[float, float]=(0, 0)) -> None:
        self.object_name = object_name
        self.bbox = bbox
        self.offset = offset
        self.conf = conf
        self.xmin = self.bbox.x1
        self.xmax = self.bbox.x2
        self.ymin = self.bbox.y1
        self.ymax = self.bbox.y2

    def abs_bbox(self) -> Rect:
        x, y = self.offset
        return Rect(self.bbox.x1 + x, self.bbox.y1 + y, w=self.bbox.w, h = self.bbox.h)

    def __str__(self) -> str:
        return f'{self.object_name},{self.conf},{self.bbox.x1},{self.bbox.y1},{self.bbox.x2},{self.bbox.y2}'

    def __repr__(self) -> str:
        return self.__str__()

class BaseDetector:
    def __init__(self, scale_factor:float=1, map_func=lambda x: x) -> None:
        self.scale_factor = scale_factor
        self.map_func = map_func
        self.nn_time = 0
        self.init_cost_func()

    def export(self):
        pass

    def load(self) -> bool:
        pass

    def get_info(self) -> dict:
        return {'k': 1, 'b': 0, 'a': 0}

    def init_cost_func(self):
        info = self.get_info()
        info['k'] *= self.scale_factor ** 2
        print(f'DNN info', info)
        self.cost_func: CostFunc = create_dnn_cost_func(info)

    def cuda(self):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError
    
    def get_nn_time(self):
        return self.nn_time
    
    def interface(self, x: torch.Tensor, offset: Tuple[float, float]=(0, 0)) -> List[List[ObjectInfo]]:
        raise NotImplementedError

    def __call__(self, x: torch.Tensor, offset: Tuple[float, float]=(0, 0)) -> List[List[ObjectInfo]]:
        start = time.time()
        ret = self.interface(x, offset)
        for i in range(len(ret)):
            for j in range(len(ret[i])):
                ret[i][j] = self.map_func(ret[i][j])
        self.nn_time += time.time() - start
        return ret