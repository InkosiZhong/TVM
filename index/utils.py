import numpy as np
import torch
from typing import Tuple, Any
from tile import TilePos, Rect, Layout
from tile import UnionSet, ultra_fast_optimal_tiling

def array2idx(arr: np.array) -> Tuple[int, TilePos]:
    t, pos = arr[0], arr[1:]
    return (t, tuple(pos))

class DNNOutputCache:
    def __init__(self, target_dnn, dataset, target_dnn_callback=lambda x: x):
        target_dnn.cuda()
        target_dnn.eval()
        self.target_dnn = target_dnn
        self.dataset = dataset
        self.target_dnn_callback = target_dnn_callback
        self.length = dataset.get_frame_num()
        self.cache = [{} for _ in range(self.length)]
        self.nb_of_invocations = 0

    def get_frame_num(self):
        return self.dataset.get_frame_num()
        
    def __len__(self):
        return self.length
    
    def infer_tile(self, frame: torch.Tensor, tile: Rect):
        self.nb_of_invocations += 1
        with torch.no_grad():
            record = frame.unsqueeze(0).cuda()
            result = self.target_dnn(record, (tile.x1, tile.y1))
        return self.target_dnn_callback(result)

    def infer(self, frame: torch.Tensor, tile: Rect, sub_layout: Layout=None):
        #return self.infer_tile(frame, tile)
        if not sub_layout or tile.area() < 1e6:
            return self.infer_tile(frame, tile)
        # adjustment
        N, M = len(sub_layout), len(sub_layout[0])
        if N == M == 1:
            return self.infer_tile(frame, tile)
        parent = UnionSet(N, M)
        for i in range(N):
            for j in range(M):
                _, has_roi = sub_layout[i][j]
                if has_roi:
                    parent[(i, j)] = [(0, 0)] # same cluster
                else:
                    parent[(i, j)] = []
        sub_layout = ultra_fast_optimal_tiling(self.target_dnn.cost_func, sub_layout, parent)
        # inference
        result = []
        N, M = len(sub_layout), len(sub_layout[0])
        for i in range(N):
            for j in range(M):
                sub_tile, has_roi = sub_layout[i][j]
                if not has_roi:
                    continue
                xmin = int(sub_tile.x1 - tile.x1)
                ymin = int(sub_tile.y1 - tile.y1)
                xmax = int(xmin + sub_tile.w)
                ymax = int(ymin + sub_tile.h)
                sub_frame = frame[:, ymin: ymax, xmin: xmax]
                result += self.infer_tile(sub_frame, sub_tile)
        return result
    
    def __getitem__(self, idx: Tuple[int, TilePos]):
        t, pos = idx
        if pos not in self.cache[t]:
            frame, tile, _ = self.dataset[idx]
            result = self.infer(frame, tile)
            self.cache[t][pos] = result
        return self.cache[t][pos]

    def __setitem__(self, idx: Tuple[int, TilePos], value: Any):
        t, pos = idx
        if pos not in self.cache[t]:
            self.cache[t][pos] = value

    def get_cache(self, 
            frame: torch.Tensor, 
            tile: Rect, 
            idx: Tuple[int, TilePos], 
            sub_layout: Layout=None
        ):
        t, pos = idx
        if pos not in self.cache[t]:
            result = self.infer(frame, tile, sub_layout)
            self.cache[t][pos] = result
        return self.cache[t][pos]

class DNNOutputCacheFloat:
    def __init__(self, target_dnn_cache, scoring_fn, idx):
        self.target_dnn_cache = target_dnn_cache
        self.scoring_fn = scoring_fn
        self.idx = idx
        
        def override_arithmetic_operator(name):
            def func(self, *args):
                value = self.target_dnn_cache[self.idx]
                value = self.scoring_fn(value)
                value = np.float32(value)
                args_f = []
                for arg in args:
                    if type(arg) is DNNOutputCacheFloat:
                        arg = np.float32(arg)
                    args_f.append(arg)
                value = getattr(value, name)(*args_f)
                return value 
            return func
        
        operator_names = [
            "__add__",
            "__sub__",
            "__mul__",
            "__truediv__", 
            "__neg__", 
            "__pos__", 
            "__radd__",
            "__rmul__",
        ]
            
        for name in operator_names:
            setattr(DNNOutputCacheFloat, name, override_arithmetic_operator(name))
        
    def __repr__(self):
        return f'DNNOutputCacheFloat(idx={self.idx})'
    
    def __float__(self):
        value = self.target_dnn_cache[self.idx]
        value = self.scoring_fn(value)
        return float(value)
    
    def get_score(self, frame: torch.Tensor, tile: Rect, sub_layout: Layout):
        value = self.target_dnn_cache.get_cache(frame, tile, self.idx, sub_layout)
        value = self.scoring_fn(value)
        return float(value)