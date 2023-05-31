import sys
sys.path.append('.')
from tile import TileFullScanDataset, create_dnn_cost_func
import numpy as np
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from blazeit.aggregation.samplers import ControlCovariateSampler
import supg.datasource as datasource
from supg.sampler import ImportanceSampler
from supg.selector import ApproxQuery
from supg.selector import RecallSelector, ImportancePrecisionTwoStageSelector
from tabulate import tabulate
from index import DNNOutputCacheFloat, array2idx
from query.miris.workload import MIRISWorkload
import time
from tqdm import tqdm
from numba import njit, jit, prange

def print_dict(
        d, header='Key', 
        ban_list = ['y_pred', 'y_true', 'source', 'ret_inds']):
    headers = [header, '']
    data = [(k,v) for k,v in d.items() if k not in ban_list]
    print(tabulate(data, headers=headers))
    print('')

@jit(parallel=True)
def propagation(y_pred, topk_reps, topk_distances):
    '''
    numba acceleration
    topk_reps here is a sub-array of the y_true
    '''
    for i in prange(len(y_pred)):
        weights = topk_distances[i]
        weights = np.sum(weights) - weights
        weights = weights / weights.sum()
        counts = topk_reps[i]
        y_pred[i] =  np.sum(counts * weights)
    return y_pred

class BaseQuery:
    def __init__(self, index):
        self.index = index
        self.df = False
        self.preprare_time = None
        self.propagation_time = None
        self.total_time = None

    def get_decode_time(self):
        return self.index.target_dnn_cache.dataset.get_decode_time()
    
    def get_nn_time(self):
        return self.index.target_dnn_cache.target_dnn.get_nn_time()

    def get_cost(self):
        return {
            'total(s)': f'{self.total_time:.2f}',
            'prepare(s)': f'{self.preprare_time:.2f}',
            'propagation(s)': f'{self.propagation_time:.2f}',
            'decode(s)': f'{self.get_decode_time():.2f}',
            'nn(s)': f'{self.get_nn_time():.2f}'
        }

    def score(self, target_dnn_output):
        raise NotImplementedError

    def propagate(self, target_dnn_cache, reps, topk_reps, topk_distances):
        if not self.df:
            t = time.time()
            y_pred = np.zeros(len(topk_reps))
            y_true = np.array(  # this array should be persistence among queries
                [DNNOutputCacheFloat(target_dnn_cache, None, array2idx(self.index.idxs[i])) 
                    for i in range(len(topk_reps))]
            )
            self.preprare_time = time.time() - t # this time should not calculated since it only need to be done once

            t = time.time()
            for i in range(len(y_true)):
                y_true[i].scoring_fn = self.score
            '''for i in tqdm(range(len(y_pred)), 'Propagation'):
                weights = topk_distances[i]
                weights = np.sum(weights) - weights
                weights = weights / weights.sum()
                counts = y_true[topk_reps[i]]
                y_pred[i] =  np.sum(counts * weights)'''
            for i in reps:
                y_true[i] = float(y_true[i])
            tmp = y_true[topk_reps]
            tmp = tmp.astype(np.float32)
            y_pred = propagation(y_pred, tmp, topk_distances)
            self.propagation_time = time.time() - t
        else:
            y_true = self.score(target_dnn_cache.df)
            y_pred = np.zeros(len(topk_reps))  
            weights = topk_distances
            weights = np.sum(weights, axis=1).reshape(-1, 1) - weights
            weights = weights / weights.sum(axis=1).reshape(-1, 1)
            counts = np.take(y_true, topk_reps)
            y_pred = np.sum(counts * weights, axis=1)
        return y_pred, y_true
    
    def _execute(self):
        raise NotImplementedError

    def execute(self, **args):
        t = time.time()
        res = self._execute(**args)
        self.total_time = time.time() - t
        res = {**args, **res, **self.get_cost()}
        print_dict(res, header=self.__class__.__name__)
        return res

class AggregateQuery(BaseQuery):
    def score(self, target_dnn_output):
        raise NotImplementedError

    def _execute(self, err_tol=0.01, confidence=0.05, batch_size=1):
        y_pred, y_true = self.propagate(
            self.index.target_dnn_cache,
            self.index.reps, self.index.topk_reps, self.index.topk_dists
        )
        r = max(1, np.amax(np.rint(y_pred)))
        sampler = ControlCovariateSampler(err_tol, confidence, y_pred, y_true, r)
        estimate, nb_samples, nb_decoded = sampler.sample(batch_size)

        res = {
            'initial_estimate': y_pred.sum(),
            'debiased_estimate': estimate,
            'nb_samples': nb_samples,
            'nb_decoded': nb_decoded,
            'y_pred': y_pred,
            'y_true': y_true
        }
        return res

    def execute_metrics(self, err_tol=0.01, confidence=0.05, batch_size=1):
        res = self._execute(err_tol, confidence, batch_size)
        res['actual_estimate'] = res['y_true'].sum() # expensive
        #print_important_result(res)
        print_dict(res, header=self.__class__.__name__)
        return res

class LimitQuery(BaseQuery):
    def score(self, target_dnn_output):
        return len(target_dnn_output)
    
    def prepare(self):
        y_pred, y_true = self.propagate(
            self.index.target_dnn_cache,
            self.index.reps, self.index.topk_reps, self.index.topk_dists
        )

        y_frame = np.zeros(len(self.index.target_dnn_cache)) # number of frames
        y_tile = [[] for _ in range(len(self.index.target_dnn_cache))]
        for i, (idx, value) in enumerate(zip(self.index.idxs, y_pred)):
            t, pos = array2idx(idx)
            y_frame[t] += value
            y_tile[t].append((i, pos))
        return y_pred, y_true, y_frame, y_tile
    
    def execute_naive_parallel(self, want_to_find, nb_to_find, GAP, batch_size):
        assert GAP == 0, 'do not support GAP>0 in this mode'
        y_pred, _, y_frame, y_tile = self.prepare()
        frame_order = np.argsort(y_frame)[::-1]
        ret_inds = []
        nb_calls = 0
        dataset = self.index.target_dnn_cache.dataset
        num_target = {}
        while len(ret_inds) < nb_to_find:
            # construct parallel decoding batch
            batch = []
            for t in frame_order:
                if y_tile[t]: # not empty
                    y_tile[t] = sorted(y_tile[t], key=lambda x: y_pred[x[0]], reverse=True)
                    y_tile[t] = [(t, pos) for (i, pos) in y_tile[t] if y_pred[i] > 0]
                    batch += y_tile[t] # all elements
                    y_tile[t] = []
                if len(batch) >= batch_size:
                    break
            ret = dataset[batch]
            for (frame, tile, sub_layout), idx in zip(ret, batch):
                t = idx[0]
                if t not in ret_inds:
                    nb_calls += 1
                    cache = self.index.target_dnn_cache.get_cache(frame, tile, idx, sub_layout)
                    if t in num_target:
                        num_target[t] += self.score(cache)
                    else:
                        num_target[t] = self.score(cache) 
                    if num_target[t] >= want_to_find:
                        ret_inds.append(t)
                        if len(ret_inds) >= nb_to_find:
                            break
        return nb_calls, ret_inds
    
    def execute_parallel(self, want_to_find, nb_to_find, GAP, batch_size):
        y_pred, _, y_frame, y_tile = self.prepare()
        frame_order = np.argsort(y_frame)[::-1]
        ret_inds = []
        visited = set() # add when the frame is decoded
        seen = set() # add when the frame is seen
        nb_calls = 0
        dataset = self.index.target_dnn_cache.dataset
        num_target = {}
        while len(ret_inds) < nb_to_find:
            # construct parallel decoding batch
            batch = []
            for t in frame_order:
                if t in seen:
                    continue
                skip = False
                for offset in range(-GAP, GAP+1):
                    if offset == 0:
                        continue
                    if t + offset in visited:
                        skip = True
                        break
                if skip:    # do this frame later
                    continue
                if t not in visited:
                    y_tile[t] = sorted(y_tile[t], key=lambda x: y_pred[x[0]], reverse=True)
                    y_tile[t] = [(t, pos) for (i, pos) in y_tile[t] if y_pred[i] > 0]
                    visited.add(t)
                if len(y_tile[t]) == 0:
                    visited.remove(t)   # remove the sheld
                    continue
                batch.append(y_tile[t].pop(0)) # the highest scored element
                if len(batch) == batch_size:
                    break
            ret = dataset[batch]
            for (frame, tile, sub_layout), idx in zip(ret, batch):
                nb_calls += 1
                t = idx[0]
                cache = self.index.target_dnn_cache.get_cache(frame, tile, idx, sub_layout)
                if t in num_target:
                    num_target[t] += self.score(cache)
                else:
                    num_target[t] = self.score(cache) 
                if num_target[t] >= want_to_find:
                    ret_inds.append(t)
                    if len(ret_inds) >= nb_to_find:
                        break
                    for offset in range(-GAP, GAP+1):
                        seen.add(t + offset)
        return nb_calls, ret_inds

    def execute_sequential(self, want_to_find, nb_to_find, GAP):
        y_pred, y_true, y_frame, y_tile = self.prepare()
        frame_order = np.argsort(y_frame)[::-1]
        ret_inds = []
        visited = set()
        nb_calls = 0
        for t in frame_order:
            if t in visited:
                continue
            tile_order = sorted(y_tile[t], key=lambda x: y_pred[x[0]], reverse=True)
            num_target = 0
            for i, _ in tile_order:
                if y_pred[i] <= 0:
                    break
                nb_calls += 1
                y_true[i] = float(y_true[i])
                num_target += y_true[i]
                if num_target >= want_to_find:
                    ret_inds.append(t)
                    for offset in range(-GAP, GAP+1):
                        visited.add(offset + t)
                    break
            if len(ret_inds) >= nb_to_find:
                break
        return nb_calls, ret_inds

    def _execute(self, want_to_find=5, nb_to_find=10, GAP=300, batch_size=1, naive_parallel=False):
        parallel = batch_size > 1
        if parallel:
            if naive_parallel:
                nb_calls, ret_inds = self.execute_naive_parallel(want_to_find, nb_to_find, GAP, batch_size)
            else:
                nb_calls, ret_inds = self.execute_parallel(want_to_find, nb_to_find, GAP, batch_size)
        else:
            nb_calls, ret_inds = self.execute_sequential(want_to_find, nb_to_find, GAP,)
            
        res = {
            'nb_calls': nb_calls,
            'nb_inds': len(ret_inds),
            'ret_inds': ret_inds
        }
        return res

    def execute_metrics(self, want_to_find=5, nb_to_find=10, GAP=300, batch_size=1):
        return self.execute(want_to_find, nb_to_find, GAP, batch_size)

class SUPGPrecisionQuery(BaseQuery):
    def score(self, target_dnn_output):
        raise NotImplementedError

    def _execute(self, budget, batch_size=1):
        y_pred, y_true = self.propagate(
            self.index.target_dnn_cache,
            self.index.reps, self.index.topk_reps, self.index.topk_dists
        )

        start = time.time()
        reps_y_pred = np.full(len(self.index.target_dnn_cache), -1.) # number of frame
        reps_idxs = np.full_like(reps_y_pred, -1, dtype=np.int32)
        for i, (idx, value) in enumerate(zip(self.index.idxs, y_pred)):
            t = idx[0]
            if value > reps_y_pred[t]:
                reps_idxs[t] = i
                reps_y_pred[t] = value
        valid_reps = np.where(reps_idxs != -1)
        reps_idxs = reps_idxs[valid_reps]
        reps_y_pred = reps_y_pred[valid_reps]
        reps_y_true = y_true[reps_idxs]
        reorganize = time.time() - start

        source = datasource.RealtimeDataSource(reps_y_pred, reps_y_true)
        sampler = ImportanceSampler()
        query = ApproxQuery(
            qtype='pt',
            min_recall=0.95, min_precision=0.95, delta=0.05,
            budget=budget
        )
        selector = ImportancePrecisionTwoStageSelector(query, source, sampler)
        inds = selector.select(batch_size)

        res = {
            'ret_inds': inds,
            'inds_length': inds.shape[0],
            'y_true': y_true,
            'y_pred': y_pred,
            'source': source,
            'reorganize(s)': f'{reorganize:.2f}'
        }

        return res

    def execute_metrics(self, budget, batch_size=1):
        res = self._execute(budget, batch_size)
        source = res['source']
        inds = res['inds']
        nb_got = np.sum(source.lookup(inds))
        nb_true = res['y_true'].sum()
        precision = nb_got / len(inds)
        recall = nb_got / nb_true
        res['precision'] = precision
        res['recall'] = recall
        print_dict(res, header=self.__class__.__name__)
        return res

class SUPGRecallQuery(SUPGPrecisionQuery):
    def _execute(self, budget, batch_size=1):
        y_pred, y_true = self.propagate(
            self.index.target_dnn_cache,
            self.index.reps, self.index.topk_reps, self.index.topk_dists
        )

        start = time.time()
        reps_y_pred = np.full(len(self.index.target_dnn_cache), -1.) # number of frame
        reps_idxs = np.full_like(reps_y_pred, -1, dtype=np.int32)
        for i, (idx, value) in enumerate(zip(self.index.idxs, y_pred)):
            t = idx[0]
            if value > reps_y_pred[t]:
                reps_idxs[t] = i
                reps_y_pred[t] = value
        valid_reps = np.where(reps_idxs != -1)
        reps_idxs = reps_idxs[valid_reps]
        reps_y_pred = reps_y_pred[valid_reps]
        reps_y_true = y_true[reps_idxs]
        reorganize = time.time() - start

        source = datasource.RealtimeDataSource(reps_y_pred, reps_y_true)
        sampler = ImportanceSampler()
        query = ApproxQuery(
            qtype='rt',
            min_recall=0.90, min_precision=0.90, delta=0.05,
            budget=budget
        )
        selector = RecallSelector(query, source, sampler, sample_mode='sqrt')
        inds = selector.select(batch_size)

        res = {
            'ret_inds': inds,
            'inds_length': inds.shape[0],
            'y_true': y_true,
            'y_pred': y_pred,
            'source': source,
            'reorganize(s)': f'{reorganize:.2f}'
        }
        return res

    def execute_metrics(self, budget, y=None):
        res = self._execute(budget, y)
        source = res['source']
        inds = res['ret_inds']
        nb_got = np.sum(source.lookup(inds))
        nb_true = res['y_true'].sum()
        precision = nb_got / len(inds)
        recall = nb_got / nb_true
        res['precision'] = precision
        res['recall'] = recall
        print_dict(res, header=self.__class__.__name__)
        return res

class TrackQuery(BaseQuery):
    def score(self, target_dnn_output):
        return len(target_dnn_output)
    
    def prepare(self, workload_path):
        y_pred, y_true = self.propagate(
            self.index.target_dnn_cache,
            self.index.reps, self.index.topk_reps, self.index.topk_dists
        )

        workloads = MIRISWorkload(workload_path)
        # sorted by idx[0], that is frame index t
        #y_pred, y_true = zip(*sorted(zip(y_pred, y_true), key=lambda x: x[1].idx[0]))
        return y_pred, y_true, workloads
    
    def execute_parallel(self, workload_path, batch_size):
        y_pred, y_true, workloads = self.prepare(workload_path)
        frames = {}
        for p, t in zip(y_pred, y_true):
            if isinstance(t, float):
                continue
            idx = t.idx[0]
            if idx in frames:
                frames[idx].append((p, t))
            else:
                frames[idx] = [(p, t)]

        cnt = 0
        fp = 0
        dataset = self.index.target_dnn_cache.dataset
        for i, workload in enumerate(tqdm(workloads, 'workloads', position=1)):
            batch = []
            for idx in tqdm(workload, f'workload {i}'):
                if len(batch) >= batch_size:
                    ret = dataset[batch]
                    for (frame, tile, sub_layout), idx in zip(ret, batch):
                        x = self.index.target_dnn_cache.get_cache(frame, tile, idx, sub_layout)
                        if self.score(x) <= 0:
                            fp += 1
                    batch = []
                if idx not in frames:
                    continue
                for p, t in frames[idx]:
                    if p <= 0:
                        continue
                    batch.append(t.idx)
                    cnt += 1
            if len(batch) > 0:
                ret = dataset[batch]
                for (frame, tile, sub_layout), idx in zip(ret, batch):
                    _ = self.index.target_dnn_cache.get_cache(frame, tile, idx, sub_layout)
                batch = []
        res = {
            'tiles': cnt,
            #'false-positive': fp
        }
        return res

    def execute_sequential(self, workload_path):
        y_pred, y_true, workloads = self.prepare(workload_path)
        frames = {}
        cnt = 0
        fp = 0
        for p, t in zip(y_pred, y_true):
            if isinstance(t, float):
                continue
            idx = t.idx[0]
            if idx in frames:
                frames[idx].append((p, t))
            else:
                frames[idx] = [(p, t)]

        dataset = self.index.target_dnn_cache.dataset
        for i, workload in enumerate(tqdm(workloads, 'workloads', position=1)):
            for idx in tqdm(workload, f'workload {i}'):
                if idx not in frames:
                    continue
                for p, t in frames[idx]:
                    if p <= 0:
                        continue
                    frame, tile, sub_layout = dataset[t.idx]
                    x = self.index.target_dnn_cache.get_cache(frame, tile, t.idx, sub_layout)
                    if self.score(x) <= 0:
                        fp += 1
                    cnt += 1
        res = {
            'tiles': cnt,
            'false-positive': fp
        }
        return res

    def _execute(self, workload_path: str, batch_size: int=1):
        parallel = batch_size > 1
        if parallel:
            res = self.execute_parallel(workload_path, batch_size)
        else:
            res = self.execute_sequential(workload_path)
        return res

class SubframeQuery:
    def __init__(self, root, classes, dec_type, max_workers=4) -> None:
        self.root = root
        self.classes = classes
        self.dec_type = dec_type
        self.max_workers = max_workers

    def execute(self):
        t = time.time()
        dataset = TileFullScanDataset(
            root=self.root,
            dec_type=self.dec_type,
            target_size=None,
            max_workers=self.max_workers,
            I_frame_only=False,
            info_filter=lambda x: any(cls in x for cls in self.classes),
        )
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False
        )
        area = 0
        for _, _, frame in tqdm(dataloader):
            area += int(frame.shape[-1] * frame.shape[-2])
        num_tile = len(dataloader)
        total_time = time.time() - t
        res = {
            'cost': total_time,
            'tiles': num_tile,
            'area': area
        }
        print_dict(res, header=self.__class__.__name__)