from typing import List, Sequence

import pandas as pd
import numpy as np


class DataSource:
    def lookup(self, idxs: Sequence) -> np.ndarray:
        raise NotImplemented()

    def filter(self, ids) -> np.ndarray:
        labels = self.lookup(ids)
        return np.array([ids[i] for i in range(len(ids)) if labels[i]])

    def get_ordered_idxs(self) -> np.ndarray:
        raise NotImplemented()

    def get_y_prob(self) -> np.ndarray:
        raise NotImplemented()

    def lookup_yprob(self, ids) -> np.ndarray:
        raise NotImplemented()


class RealtimeDataSource(DataSource):
    def __init__(
        self,
        y_pred,
        y_true,
        seed=123041,
    ):
        self.y_pred = y_pred
        self.y_true = y_true
        self.dataset = None # for batch processing
        for yt in y_true:
            if not isinstance(yt, float):
                self.dataset = yt.target_dnn_cache.dataset
                break
        self.random = np.random.RandomState(seed)
        self.proxy_score_sort = np.lexsort((self.random.random(y_pred.size), y_pred))[::-1]
        self.lookups = 0

    def generate_batch(self, ids):
        batch_t, batch = [-1 for _ in range(len(ids))], []
        for i, t in enumerate(ids):
            yt = self.y_true[t]
            if not isinstance(yt, float):
                batch_t[i] = len(batch)
                batch.append(yt.idx)
        ret = self.dataset[batch]
        return batch_t, ret

    def lookup(self, ids, batch_size=1):
        #y_true = self.y_true.copy()
        if batch_size > 1:
            batch_t, ret = None, None
            for i, t in enumerate(ids):
                if i % batch_size == 0:
                    end = min(i + batch_size, len(ids))
                    batch_t, ret = self.generate_batch(ids[i:end])
                if not isinstance(self.y_true[t], float):
                    frame, tile, sub_layout = ret[batch_t[i % batch_size]]
                    self.y_true[t] = self.y_true[t].get_score(frame, tile, sub_layout)
        self.lookups += len(ids)
        #err = np.abs(self.y_true[ids].astype(np.float) - self.y_pred[ids].astype(np.float))
        #print(np.mean(err), np.std(err))
        #wrong = [y_true[i].idx for i in ids if self.y_true[i] == 0 and self.y_pred[i] > 0.5 and type(y_true[i]) != float]
        #wrong = [None for yt, yp in zip(self.y_true[ids], self.y_pred[ids]) if yp > 0.5 and yt == 0]
        #print(len(wrong))
        '''for idx in wrong:
            frame, _ = self.dataset[idx]
            from torchvision import utils as vutils
            vutils.save_image(frame, f'out/img/{idx[0]}_{idx[1]}.png')
        exit()'''
        return self.y_true[ids].astype(np.float)

    def get_ordered_idxs(self) -> np.ndarray:
        return self.proxy_score_sort

    def get_y_prob(self) -> np.ndarray:
        return self.y_pred[self.proxy_score_sort]

    def lookup_yprob(self, ids) -> np.ndarray:
        return self.y_pred[ids]


class DFDataSource(DataSource):
    def __init__(
            self,
            df,
            drop_p=None,
            seed=123041
    ):
        self.random = np.random.RandomState(seed)
        if drop_p is not None:
            pos = df[df['label'] == 1]
            remove_n = int(len(pos) * drop_p)
            drop_indices = self.random.choice(pos.index, remove_n, replace=False)
            df = df.drop(drop_indices).reset_index(drop=True)
            df.id = df.index

        print(len(df[df['label'] == 1]) / len(df))
        self.df_indexed = df.set_index(["id"])
        self.df_sorted = df.sort_values(
                ["proxy_score"], axis=0, ascending=False).reset_index(drop=True)
        self.lookups = 0

    def lookup(self, ids):
        self.lookups += len(ids)
        return self.df_indexed.loc[ids]["label"].values

    def get_ordered_idxs(self) -> np.ndarray:
        return self.df_sorted["id"].values

    def get_y_prob(self) -> np.ndarray:
        return self.df_sorted["proxy_score"].values

    def lookup_yprob(self, ids) -> np.ndarray:
        return self.df_indexed.loc[ids]['proxy_score'].values
