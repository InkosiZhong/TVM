import math
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.linear_model import LinearRegression
import time

class Sampler(object):
    def __init__(self, err_tol, conf, Y_pred, Y_true, R):
        self.err_tol = err_tol
        self.conf = conf
        self.Y_true = Y_true
        self.Y_pred = Y_pred
        self._Y_true = None
        self._Y_pred = None
        self.dataset = None
        self.target_dnn = None
        for yt in Y_true:
            if not isinstance(yt, float):
                self.dataset = yt.target_dnn_cache.dataset
                self.target_dnn = yt.target_dnn_cache.target_dnn
                break
        self.nb_decoded = 0

        self.R = R

    def get_sample(self, Y_pred, Y_true, nb_samples):
        raise NotImplementedError

    def reset(self, Y_pred, Y_true):
        pass
    
    def reestimate(self, Y_pred, Y_true, nb_samples):
        return None, None

    def permute(self, Y_pred, Y_true):
        #Y_pred, Y_true = zip(*[(p, t) for p, t in zip(Y_pred, Y_true) if p > 0])
        #Y_pred, Y_true = np.array(Y_pred), np.array(Y_true)
        p = np.random.permutation(len(Y_pred))
        Y_pred, Y_true = Y_pred[p], Y_true[p]
        return Y_pred, Y_true
    
    def generate_batch(self, round, batch_size):
        batch_t, batch = [-1 for _ in range(batch_size)], []
        for i, y_true in enumerate(self._Y_true[round*batch_size: (round+1)*batch_size]):
            if not isinstance(y_true, float):
                batch_t[i] = len(batch)
                batch.append(y_true.idx)
        self.nb_decoded += len(batch)
        ret = self.dataset[batch]
        return batch_t, ret
    
    def pre_sample(self, nb_samples, ret):
        self._Y_true[nb_samples] = self._Y_true[nb_samples].get_score(*ret)

    def sample(self, batch_size):
        parallel = batch_size > 1
        #Y_pred, Y_true = self.permute(self.Y_pred.astype(np.float32), self.Y_true.astype(np.float32))
        self._Y_pred, self._Y_true = self.permute(self.Y_pred.astype(np.float32), self.Y_true)
        '''idx = (181894, (1, 1))
        frame, tile, sub_layout = self.dataset[idx]
        self._Y_true[0].target_dnn_cache.get_cache(frame, tile, idx, sub_layout)
        exit()'''
        if parallel:
            batch_t, ret = self.generate_batch(0, batch_size)
            for i in range(len(batch_t)):
                if batch_t[i] == -1:
                    continue
                self.pre_sample(i, ret[batch_t[i]])
        self.reset(self._Y_pred, self._Y_true)

        LB = -10000000
        UB = 10000000
        t = 1
        k = 1
        beta = 1.5
        R = self.R
        eps = self.err_tol
        p = 1.1
        c = self.conf * (p - 1) / p

        if parallel and not isinstance(self._Y_true[0], float):
            self.pre_sample(t, ret[batch_t[0]])
        Xt_sum = self.get_sample(self._Y_pred, self._Y_true, t)
        Xt_sqsum = Xt_sum * Xt_sum
        # while (1 + eps) * LB < (1 - eps) * UB:
        while LB + eps < UB - eps:
            t += 1
            if parallel and t % batch_size == 0:
                batch_t, ret = self.generate_batch(t // batch_size, batch_size)
            if parallel and not isinstance(self._Y_true[t], float):
                self.pre_sample(t, ret[batch_t[t % batch_size]])
            if t > 0 and t % 1000 == 0:
                print(f'[{t}] decode: {self.dataset.get_decode_time():.2f}s, nn: {self.target_dnn.get_nn_time():.2f}s')
            if t > np.floor(beta ** k):
                k += 1
                alpha = np.floor(beta ** k) / np.floor(beta ** (k - 1))
                dk = c / (math.log(k, p) ** p)
                x = -alpha * np.log(dk) / 3
                t1, t2 = self.reestimate(self._Y_pred, self._Y_true, t)
                if t1 is not None and t2 is not None:
                    Xt_sum = t1
                    Xt_sqsum = t2
            sample = self.get_sample(self._Y_pred, self._Y_true, t)
            Xt_sum += sample
            Xt_sqsum += sample * sample
            Xt = Xt_sum / t
            sigmat = np.sqrt(1/t * (Xt_sqsum - Xt_sum ** 2 / t))
            # Finite sample correction
            sigmat *= np.sqrt((len(self._Y_true) - t) / (len(self._Y_true) - 1))

            ct = sigmat * np.sqrt(2 * x / t) + 3 * R * x / t
            LB = max(LB, np.abs(Xt) - ct)
            UB = min(UB, np.abs(Xt) + ct)

        estimate = np.sign(Xt) * 0.5 * \
            ((1 + eps) * LB + (1 - eps) * UB)
        if not parallel:
            self.nb_decoded = t
        return estimate * len(self._Y_true), t, self.nb_decoded


class TrueSampler(Sampler):
    def get_sample(self, Y_pred, Y_true, nb_samples):
        return Y_true[nb_samples]


class ControlCovariateSampler(Sampler):
    def __init__(self, *args):
        super().__init__(*args)
        self.tau = np.mean(self.Y_pred)
        self.var_t = np.var(self.Y_pred)

    def reset(self, Y_pred, Y_true):
        _Y_true = Y_true[0:100].astype(np.float32)
        self._Y_true[0:100] = _Y_true
        self.cov = np.cov(_Y_true, Y_pred[0:100])[0][1]
        self.c = -1 * self.cov / self.var_t

    def reestimate(self, Y_pred, Y_true, nb_samples):
        yt_samp = Y_true[0:nb_samples].astype(np.float32)
        self._Y_true[0:nb_samples] = yt_samp
        yp_samp = Y_pred[0:nb_samples]
        self.cov = np.cov(yt_samp, yp_samp)[0][1]
        self.c = -1 * self.cov / self.var_t

        samples = yt_samp + self.c * (yp_samp - self.tau)
        Xt_sum = np.sum(samples)
        Xt_sqsum = sum([x * x for x in samples])
        return Xt_sum, Xt_sqsum

    def _get_yp(self, Y_true, Y_pred, nb_samples):
        return Y_pred[nb_samples]

    def get_sample(self, Y_pred, Y_true, nb_samples):
        x = float(self._Y_true[nb_samples])
        self._Y_true[nb_samples] = x
        yt_samp = self._Y_true[nb_samples]
        yp_samp = self._get_yp(Y_true, Y_pred, nb_samples)

        sample = yt_samp + self.c * (yp_samp - self.tau)
        return sample


class MultiControlCovariateSampler(ControlCovariateSampler):
    def __init__(self, *args):
        super().__init__(*args)

    def reset(self, Y_pred, Y_true):
        self.reg = LinearRegression().fit(Y_pred[0:100], Y_true[0:100])
        yp = self.reg.predict(Y_pred[0:100])
        super().reset(yp, Y_true)

    def reestimate(self, Y_pred, Y_true, nb_samples):
        self.reg = LinearRegression().fit(Y_pred[0:nb_samples], Y_true[0:nb_samples])
        yp = self.reg.predict(Y_pred[0:nb_samples])
        return super().reestimate(yp, Y_true, nb_samples)

    def _get_yp(self, Y_true, Y_pred, nb_samples):
        return self.reg.predict(Y_pred[nb_samples:nb_samples + 1])[0]
