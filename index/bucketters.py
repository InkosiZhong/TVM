import numpy as np
import tqdm
from numba import njit, jit, prange
np.set_printoptions(threshold=np.inf)

@njit(parallel=True)
def get_dists(x, embeddings):
    dists = np.zeros(len(embeddings), dtype=np.float32)
    for i in prange(len(embeddings)):
        dists[i] = np.sqrt(np.sum((x - embeddings[i]) ** 2))
    return dists

@njit(parallel=True)
def get_and_update_dists(x, embeddings, min_dists):
    dists = np.zeros(len(embeddings), dtype=np.float32)
    for i in prange(len(embeddings)):
        dists[i] = np.sqrt(np.sum((x - embeddings[i]) ** 2))
        if dists[i] < min_dists[i]:
            min_dists[i] = dists[i]
    return dists

class Bucketter(object):
    def __init__(
            self,
            nb_buckets: int,
            seed: int=123456
    ):
        self.nb_buckets = nb_buckets
        self.rand = np.random.RandomState(seed)

    def topk(self, k, dists):
        topks = []
        for i in tqdm.tqdm(range(len(dists)), f'Top-{5}'):
            topks.append(np.argpartition(dists[i], k)[0:k]) # select topk
        return np.array(topks)

NON_CACHE_THRESHOLD = 1e6
class FPFRandomBucketter(Bucketter):
    def bucket(
            self,
            embeddings: np.ndarray,
            max_k: int,
            percent_fpf: float=0.75,
    ):
        reps = np.full(self.nb_buckets, -1)
        reps[0] = self.rand.randint(len(embeddings)) # random select the first representation index
        min_dists = np.full(len(embeddings), np.Inf, dtype=np.float32)

        # get the top_k representations for each embedding
        topk_reps = np.zeros((len(embeddings), max_k), dtype=np.int32)
        topk_dists = np.zeros((len(embeddings), max_k)) # (N, k)

        if len(embeddings) > NON_CACHE_THRESHOLD:
            print(f'embeddings is too large ({len(embeddings)}), use non-cache mode')
            # avoid memory overflow
            _ = get_and_update_dists(
                embeddings[reps[0]], embeddings, min_dists
            )
        else:
            dists = np.zeros((self.nb_buckets, len(embeddings))) # (M, N)
            # distances between the first representation and the embeddings
            dists[0, :] = get_and_update_dists(
                embeddings[reps[0]], embeddings, min_dists
            )

        # random selection part
        num_random = int((1-percent_fpf)*len(reps))
        for i in tqdm.tqdm(range(1, num_random), desc='RandomBucketter'):
            reps[i] = self.rand.randint(len(embeddings))
            if len(embeddings) > NON_CACHE_THRESHOLD:
                _ = get_and_update_dists(
                    embeddings[reps[i]], embeddings, min_dists
                )
            else:
                dists[i, :] = get_and_update_dists(
                    embeddings[reps[i]], embeddings, min_dists
                )

        # FPF selection part
        for i in tqdm.tqdm(range(num_random, self.nb_buckets), desc='FPFBucketter'):
            reps[i] = np.argmax(min_dists) # get the farest point as a representation
            if len(embeddings) > NON_CACHE_THRESHOLD:
                _ = get_and_update_dists(
                    embeddings[reps[i]], embeddings, min_dists
                )
            else:
                dists[i, :] = get_and_update_dists(
                    embeddings[reps[i]], embeddings, min_dists
                )

        if len(embeddings) > NON_CACHE_THRESHOLD:
            # avoid memory overflow
            for i in tqdm.tqdm(range(len(embeddings)), 'one-by-one'):
                dists = get_dists(embeddings[i], embeddings[reps]) # (M, )
                topk_reps[i] = np.argpartition(dists, max_k)[0:max_k]
                topk_dists[i] = dists[topk_reps[i]]
        else:
            dists = dists.transpose() # (N, M)
            topk_reps = self.topk(max_k, dists)
            for i in tqdm.tqdm(range(len(topk_dists)), f'Top-{max_k} dists'):
                topk_dists[i] = dists[i, topk_reps[i]]
        for i in tqdm.tqdm(range(len(topk_reps)), f'Top-{max_k} reps'):
            topk_reps[i] = reps[topk_reps[i]]
        return reps, topk_reps, topk_dists
    
class CrackingBucketter(Bucketter):
    def bucket(
        self,
        embeddings: np.ndarray,
        max_k: int,
        idxs: list
    ):
        reps = idxs
        min_dists = np.full(len(embeddings), np.Inf, dtype=np.float32)
        dists = np.zeros((len(reps), len(embeddings)), dtype=np.float32)
        for i in tqdm.tqdm(range(len(reps)), desc='Cracking'):
            dists[i, :] = get_and_update_dists(
                    embeddings[reps[i]],
                    embeddings,
                    min_dists
            )
        dists = dists.transpose()
        topk_reps = self.topk(max_k, dists)
        topk_dists = np.zeros([len(topk_reps), max_k])
        
        for i in range(len(topk_dists)):
            topk_dists[i] = dists[i, topk_reps[i]]
        for i in range(len(topk_reps)):
            topk_reps[i] = reps[topk_reps[i]]
            
        return reps, topk_reps, topk_dists