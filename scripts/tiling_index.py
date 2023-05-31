import sys
import json
import torch
import torchvision
from typing import List
import cv2
import torch
from torchvision.transforms.functional import to_tensor
sys.path.append('.')
from codec import VPFDecoder, CV2Decoder
from tile import *
from tile.utils import LRUBuffer, tuple2str, union_join, union_find
from index import Index, IndexConfig
from query.detector import YoloV5, MaskRCNN
import argparse
from scripts.script_utils import show_args, show_config
DEC_TYPE = VPFDecoder

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
    
class TilingIndex(Index):
    def __init__(self, config: IndexConfig, path: str, labels: list, threshold: float):
        self.train_data_path = path
        self.labels = labels
        self.threshold = threshold
        super().__init__(config)

    def get_target_dnn(self):
        model = YoloV5(model_capacity='yolov5m').cuda()
        return model
        
    def get_embedding_dnn(self):
        model = torchvision.models.resnet18(pretrained=True, progress=True)
        model.fc = torch.nn.Linear(512, 128)
        model = model
        return model
    
    def get_pretrained_embedding_dnn(self):
        model = torchvision.models.resnet18(pretrained=True, progress=True)
        model.fc = torch.nn.Identity()
        return model
    
    def get_target_dnn_dataset(self, train=True, random_select=False):
        assert train, 'TilingIndex cannot be used for testing'
        if random_select:
            dataset = TileRandomSelectDataset(
                self.train_data_path,
                DEC_TYPE, 
                target_size=None, #(224, 224),
                fast_seek=True,
                codec_cache_size=0,
                frame_cache_size=0,
                cache_update_alg=LRUBuffer,
                cache_unhit=False
            )
        else:
            dataset = TileFullScanDataset(
                self.train_data_path,
                DEC_TYPE, 
                target_size=None, #(224, 224),
                max_workers=4,
                I_frame_only=False
            )
        return dataset
    
    def get_embedding_dnn_dataset(self, train=True, random_select=False):
        assert train, 'TilingIndex cannot be used for testing'
        if random_select:
            dataset = TileRandomSelectDataset(
                self.train_data_path,
                DEC_TYPE, 
                target_size=(224, 224),
                fast_seek=True,
                codec_cache_size=0,
                frame_cache_size=0,
                cache_update_alg=LRUBuffer,
                cache_unhit=False
            )
        else:
            dataset = TileFullScanDataset(
                self.train_data_path,
                DEC_TYPE, 
                target_size=(224, 224),
                max_workers=4,
                I_frame_only=False
            )
        return dataset
    
    def target_dnn_callback(self, target_dnn_output):
        label = []
        for output in target_dnn_output[0]:
            if output.conf > self.threshold and output.object_name in self.labels:
                label.append(output)
        return label
        
    def is_close(self, label1, label2):
        def mapping(obj):
            if obj == 'truck':
                return 'car'
            return obj
        objects1 = set([mapping(obj.object_name) for obj in label1])
        objects2 = set([mapping(obj.object_name) for obj in label2])
        return objects1 == objects2
        #return objects1.issubset(objects2) or objects2.issubset(objects1)
        
    def init(self):
        torch.cuda.empty_cache()
        self.do_mining()
        torch.cuda.empty_cache()
        self.do_training()

class TilingConfig(IndexConfig):
    def __init__(self):
        super().__init__()
        self.do_mining = True
        self.do_training = True
        #self.do_infer = False
        #self.do_bucketting = False
        
        self.batch_size = 16
        self.nb_train = 3000
        self.train_margin = 1.0
        self.train_lr = 1e-4
        self.max_k = 5
        self.nb_buckets = 7000
        self.nb_training_its = 12000


def get_dense_tile(frames: List[Rect], tile: Rect, seg_roi: List[List[Rect]]) -> torch.Tensor:
    '''
    return a tiled frame with the most dense ROIs
    '''
    select_frame = frames[0]
    max_area = 0
    for frame, roi_list in zip(frames, seg_roi):
        #roi_list = [intersect_rect(roi, tile) for roi in roi_list]
        #area = sum([roi.area() for roi in roi_list if roi is not None])
        area = 0
        for roi in roi_list:
            if tile.contain(roi.ctr()):
                area += roi.area()
        if area > max_area:
            select_frame = frame
            max_area = area
    x1, y1, x2, y2 = tile.xyxy()
    x = select_frame[int(y1):int(y2), int(x1):int(x2), :].copy()
    x = cv2.resize(x, (224, 224), interpolation=cv2.INTER_NEAREST) # fastest
    x = to_tensor(x).unsqueeze(0).cuda()
    return x

def clustering(layout, threshold):
    N, M = len(layout), len(layout[0]) # rows and cols
    parent = UnionSet(N, M) # union set
    # we first create a union-set using the semantic similarity
    for i1 in range(N):
        for j1 in range(M):
            _, has_roi1, idx1 = layout[i1][j1] 
            if not has_roi1: # find a roi tile
                continue
            for i2 in range(i1, N):
                for j2 in range(j1+1 if i1==i2 else 0, M):
                    _, has_roi2, idx2 = layout[i2][j2]
                    if not has_roi2: # find a roi tile
                        continue
                    sim = torch.cosine_similarity(idx1, idx2).item()
                    #sim = np.dot(idx1[0], idx2[0]) / (np.linalg.norm(idx1[0]) * np.linalg.norm(idx2[0]))
                    if sim > threshold:
                        union_join((i1, j1), (i2, j2), parent)
    for i in range(N):
        for j in range(M):
            _, has_roi, _ = layout[i][j]
            if has_roi:
                p = union_find((i, j), parent)
                parent[(i, j)] = p
            else:
                parent[(i, j)] = None
    for i in range(N):
        for j in range(M):
            if parent[(i, j)]:
                parent[(i, j)] = [parent[(i, j)]]
            else:
                parent[(i, j)] = []
    return parent

def semantic_tiling(index, cost_func, frames, layout, seg_roi, threshold, greedy=False):
    N, M = len(layout), len(layout[0]) # rows and cols
    # add embedding
    batch = []
    semantic_layout = [[None for _ in range(M)] for _ in range(N)]
    for i in range(N):
        for j in range(M):
            tile, has_roi = layout[i][j]
            if has_roi:
                x = get_dense_tile(frames, tile, seg_roi)
                k = len(batch)
                batch.append(x)
            else:
                k = None
            semantic_layout[i][j] = (tile, has_roi, k)
    if len(batch) > 0:
        batch = torch.cat(batch, dim=0)
        with torch.no_grad():
            batch = index.embedding_dnn(batch)
        for i, line in enumerate(semantic_layout):
            for j, (tile, has_roi, k) in enumerate(line):
                if k is not None:
                    idx = batch[k].unsqueeze(0)#.cpu().numpy()
                    semantic_layout[i][j] = (tile, has_roi, idx)
        parent = clustering(semantic_layout, threshold)
        # tiling
        if greedy:
            #layout = greedy_tiling(cost_func, layout, parent)
            layout = fast_greedy_tiling(cost_func, layout, parent)
        else:
            #layout = optimal_tiling(cost_func, layout, parent)
            #layout = fast_optimal_tiling(cost_func, layout, parent)
            #layout = very_fast_optimal_tiling(cost_func, layout, parent)
            layout = ultra_fast_optimal_tiling(cost_func, layout, parent)
    return layout

def cluster_non_roi(arr):
    cluster = {}
    tmp = float('inf')
    key = None
    for x in arr:
        if x - tmp != 1:
            key = x
            cluster[key] = 1
        else:
            cluster[key] += 1
        tmp = x
    return cluster

def hierarchical_layout(raw_layout, layout):
    N1, M1 = len(layout), len(layout[0])
    N2, M2 = len(raw_layout), len(raw_layout[0])
    for i1 in range(N1):
        for j1 in range(M1):
            tile1, has_roi1 = layout[i1][j1]
            if not has_roi1:
                continue
            '''layout[i1][j1] = (tile1, has_roi1, {
                'rows': 1, 
                'cols': 1,
                'roi': {'0_0': str(tile1)}
            })
            continue'''
            sub_layout = []
            ii, jj = -1, -1
            row, col = 0, 0
            for i2 in range(N2):
                for j2 in range(M2):
                    tile2, has_roi2 = raw_layout[i2][j2]
                    if tile1.include(tile2):
                        if ii < 0 or jj < 0:
                            ii, jj = i2, j2
                        row, col = i2 - ii + 1, j2 - jj + 1
                        sub_layout.append((tile2, has_roi2))
            if len(sub_layout) != row * col:
                print(f'error length of sub-layout ({len(sub_layout)}) does not match {row}x{col}')
            assert len(sub_layout) == row * col, f'length of sub-layout ({len(sub_layout)}) does not match {row}x{col}'
            non_roi_row = [i for i in range(row)]
            non_roi_col = [j for j in range(col)]
            for i in range(row):
                for j in range(col):
                    _, has_roi = sub_layout[i * col + j]
                    if has_roi:
                        if i in non_roi_row:
                            non_roi_row.remove(i)
                        if j in non_roi_col:
                            non_roi_col.remove(j)
            non_roi_row = cluster_non_roi(non_roi_row)
            non_roi_col = cluster_non_roi(non_roi_col)
            # format sublayout
            roi = {}
            for i in range(row):
                for j in range(col):
                    tile2, has_roi2 = sub_layout[i * col + j]
                    if has_roi2:
                        ii = i - sum([v-1 for k, v in non_roi_row.items() if k < i])
                        jj = j - sum([v-1 for k, v in non_roi_col.items() if k < j])
                        roi[tuple2str((ii, jj))] = str(tile2)
            for v in non_roi_row.values():
                row -= v - 1
            for v in non_roi_col.values():
                col -= v - 1
            layout[i1][j1] = (tile1, has_roi1, {
                'rows': row, 
                'cols': col,
                'roi': roi
            })
    return layout

def simple_merging(layout, N, M, tile_pair):
    # create new layout
    # horizontal: combine with the i1 row
    (i1, j1), (i2, j2) = tile_pair
    for j in range(0, M):
        tile1, has_roi1 = layout[i1][j]
        for i in range(i1+1, i2+1):
            tile2, has_roi2 = layout[i][j]
            has_roi1 = has_roi1 or has_roi2
            if i == i2:
                tile1 = merge_rect(tile1, tile2)
        layout[i1][j] = (tile1, has_roi1)
    for _ in range(i1+1, i2+1):
        layout.pop(i1+1) # pop the same line since the list is changed
    # vertical: combine with the j1 col
    for i in range(0, N - (i2 - i1)):
        tile1, has_roi1 = layout[i][j1]
        for j in range(j1+1, j2+1):
            tile2, has_roi2 = layout[i][j]
            has_roi1 = has_roi1 or has_roi2
            if j == j2:
                tile1 = merge_rect(tile1, tile2)
        layout[i][j1] = (tile1, has_roi1)
    for _ in range(j1+1, j2+1):
        for i in range(len(layout)):
            layout[i].pop(j1+1)
    return layout

def merge_all(layout):
    N, M = len(layout), len(layout[0]) # rows and cols
    min_i, min_j = N, M
    max_i, max_j = 0, 0
    for i, line in enumerate(layout):
        for j, (_, has_roi) in enumerate(line):
            if has_roi:
                if i < min_i:
                    min_i = i
                if i > max_i:
                    max_i = i
                if j < min_j:
                    min_j = j
                if j > max_j:
                    max_j = j
    if max_i >= min_i and max_j >= min_j:
        tile_pair = ((min_i, min_j), (max_i, max_j))
        layout = simple_merging(layout, N, M, tile_pair)
    return layout

parser = argparse.ArgumentParser(description='Tiling Index')
parser.add_argument('--input', '-i', type=str, required=True, help='source tiled video')
parser.add_argument('--output', '-o', type=str, required=True, help='path to cache')
parser.add_argument('--config', '-c', type=str, required=True, help='path to configuration')

if __name__ == '__main__':
    args = parser.parse_args()
    show_args(parser.description, args)
    data = args.input
    cfg = json.load(open(args.config, 'r'))
    cache = args.output
    show_config(args.config, cfg)
    idx_cfg = TilingConfig()
    idx_cfg.cache_root = cache
    idx_cfg.do_mining = True
    idx_cfg.do_training = True
    index = TilingIndex(idx_cfg, data, cfg['classes'], 0.5)
    index.init()