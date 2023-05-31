import torch
import numpy as np
from tqdm.autonotebook import tqdm
import pandas as pd
from collections import defaultdict
from typing import Tuple, List
import sys
sys.path.append('.')
from tile import TilePos
from tile.utils import str2tuple

'''
LabelDataset loads the target dnn .csv files and allows you to access the target dnn outputs of given frames.
'''
class LabelDataset:
    def __init__(self, labels_fp: str, length: int, labels: List[str]):
        self.length = length
        df = pd.read_csv(labels_fp)
        df = df[df['object_name'].isin(labels)]
        frame_to_rows = defaultdict(list)
        for row in df.itertuples():
            frame_to_rows[row.frame].append(row)
        self.labels = []
        cnt = 0
        for frame_idx in range(length):
            records = frame_to_rows[frame_idx]
            tile_map = defaultdict(list)
            if len(records) > 0:
                for record in records:
                    tile = str2tuple(record[-1])
                    tile_map[tile].append(record)
                    cnt += 1
            self.labels.append(tile_map)

    def get_label_by_t(self, t: int):
        return self.labels[t]
    
    def __getitem__(self, idx: Tuple[int, TilePos]):
        t, pos = idx
        return self.labels[t][pos]

    def __len__(self):
        return self.length

class TripletDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset,            # random select dataset
            target_dnn_cache,   # labels
            list_of_idxs,       # training idx
            is_close_fn,
            length=1000
    ):
        self.dataset = dataset
        self.target_dnn_cache = target_dnn_cache
        self.list_of_idxs = list_of_idxs
        self.is_close_fn = is_close_fn
        self.length = length

        self.buckets = [] # [Bucket[idx]], a bucket has all reps that are 'close'
        for idx in tqdm(self.list_of_idxs, desc="Triplet Dataset Init"):
            t, pos = idx[0], tuple(idx[1:])
            idx = (t, pos)
            label = self.target_dnn_cache[idx]
            found = False
            for bucket in self.buckets: # try to append current rep into a bucket
                rep_idx = bucket[0]
                rep = self.target_dnn_cache[rep_idx]
                if self.is_close_fn(label, rep):
                    bucket.append(idx)
                    found = True
                    break
            if not found:
                self.buckets.append([idx]) # create new bucket

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        rand = np.random.RandomState(seed=idx)
        rand.randint(0, 100, size=10)
        
        def get_triplet_helper():
            anchor_bucket_idx = rand.randint(0, len(self.buckets)) # choose a bucket
            anchor_bucket = self.buckets[anchor_bucket_idx]
            negative_bucket_idx = rand.choice(
                    [idx for idx in range(len(self.buckets)) if idx != anchor_bucket_idx]
            ) # randomly select a DIFFERENT bucket as neg sample
            negative_bucket = self.buckets[negative_bucket_idx]

            anchor_idx = anchor_bucket[rand.choice(len(anchor_bucket))]
            positive_idx = anchor_bucket[rand.choice(len(anchor_bucket))]
            negative_idx = negative_bucket[rand.choice(len(negative_bucket))]

            return anchor_idx, positive_idx, negative_idx

        anchor_idx, positive_idx, negative_idx = get_triplet_helper()
        for i in range(200): # try to find a triplet that anchor and pos sample are not that close yet
            if abs(anchor_idx[0] - positive_idx[0]) > 30:
                break
            else:
                anchor_idx, positive_idx, negative_idx = get_triplet_helper()
        
        anchor, _ = self.dataset[anchor_idx]
        positive, _ = self.dataset[positive_idx]
        negative, _ = self.dataset[negative_idx]
        
        return anchor, positive, negative # all images
