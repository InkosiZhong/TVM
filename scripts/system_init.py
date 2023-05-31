import os, sys
sys.path.append('.')
from codec import VPFDecoder, CV2Decoder
from tile import *
from tile.utils import tuple2str
import json, time
import torch
import numpy as np
import argparse
import random
from script_utils import show_args
from tabulate import tabulate

def res2area(res_arr: np.array, min: int, step: int, sample_min: int=0, sample_max: int=np.inf):
    xs, ys = [], []
    for i, line in enumerate(res_arr):
        h = i * step + min
        if h < sample_min:
            continue
        if h > sample_max:
            break
        for j, item in enumerate(line):
            w = j * step + min
            if w < sample_min:
                continue
            if w > sample_max:
                break
            area = h * w
            xs.append(area)
            ys.append(1000 * item)
    return np.array(xs), np.array(ys)

def R_square(x, y, coef):
    y_mean = np.mean(y)
    y_fit = np.polyval(coef, x)
    return 1 - np.sum((y - y_fit) ** 2) / np.sum((y - y_mean) ** 2)

def get_codec_info(logs: str, codec: str, min: int, step: int, sample_min: int, sample_max: int):
    info = {}
    for key in ['init', 'dec', 'skip', 'rel']:
        try:
            log = np.load(os.path.join(logs, f'{codec}_{key}.npy'))
        except:
            raise RuntimeError('run test/codec_test.py first to generate logs for codec')
        xs, ys = res2area(log, min, step, sample_min, sample_max)
        k, b = np.polyfit(xs, ys, 1)
        print(f'{key} R^2={R_square(xs, ys, (k, b))}')
        info[f'k_{key}'] = k
        info[f'b_{key}'] = b
    info[f'k_seek'] = 0
    info[f'b_seek'] = 0
    return info

def get_roi_rand_dec_pred_time(codec: dict, num_pix: int, n_skip: int):
    t_init = codec['k_init'] * num_pix + codec['b_init']
    t_skip = (codec['k_skip'] * num_pix + codec['b_skip']) * n_skip
    t_dec = codec['k_dec'] * num_pix + codec['b_dec']
    t_release = codec['k_rel'] * num_pix + codec['b_rel']
    return t_init + t_skip + t_dec + t_release, t_init, t_skip, t_dec, t_release

def get_roi_rand_dec_time(path: str, i: int):
    assert i > 1, f'i={i} cannot evaluate time consumption'
    t = time.time()
    dec = VPFDecoder(path, frame_type=torch.Tensor)
    dec.skip_frame()
    t_init = (time.time() - t) * 1000
    t = time.time()
    for _ in range(i-1):
        dec.skip_frame()
    t_skip = (time.time() - t) * 1000
    t = time.time()
    dec.get_frame()
    t_dec = (time.time() - t) * 1000
    t = time.time()
    dec = None
    t_release = (time.time() - t) * 1000
    t_init -= t_skip / (i - 1)
    t_skip += t_skip / (i - 1)
    return t_init + t_skip + t_dec + t_release, t_init, t_skip, t_dec, t_release

def get_err(pred, real):
    return [(p - r) / r * 100 for p, r in zip(pred, real)]

def eval_codec(src: str, codec: str, rounds: int):
    os.system('sh scripts/clean_cache.sh')
    if codec == 'vpf':
        dec_type = VPFDecoder
    elif codec == 'cv2':
        dec_type = CV2Decoder
    else:
        raise RuntimeError(f'Unsupport codec type: {codec}')
    dataset = TileRandomSelectDataset(
        src, 
        dec_type=dec_type, 
        target_size=None,
        fast_seek=False, 
        codec_cache_size=0, 
        frame_cache_size=0,
        max_workers=1
    )
    codec_info = json.load(open('codec.json', 'r'))[codec]
    proxy = dataset.proxy
    gop = proxy.get_gop()
    segments = proxy.get_segments()
    real_cost, real_init_cost, real_skip_cost, real_dec_cost, real_rel_cost = [0] * 5
    pred_cost, pred_init_cost, pred_skip_cost, pred_dec_cost, pred_rel_cost = [0] * 5
    for j in range(rounds+1):
        while True:
            seg = random.sample(segments, 1)[0]
            all_pos = proxy.get_tile_pos(seg)
            if all_pos:
                #os.system('sh scripts/clean_cache.sh')
                pos = random.sample(all_pos, 1)[0]
                tile = proxy.get_tile_rect(seg, pos)
                path = os.path.join(src, f'{tuple2str(seg)}/{tuple2str(pos)}.mp4')
                '''start, end = seg
                i = random.randint(start, end-1)
                t = time.time()
                _, tile, _ = dataset[(i, pos)]
                cost = (time.time() - t) * 1000'''
                i = random.randint(2, gop-1)
                real_t, real_init, real_skip, real_dec, real_rel = \
                    get_roi_rand_dec_time(path, i)
                if j > 0:
                    pred_t, pred_init, pred_skip, pred_dec, pred_rel = \
                        get_roi_rand_dec_pred_time(codec_info, tile.area(), i)
                    real_cost += real_t
                    real_init_cost += real_init
                    real_skip_cost += real_skip
                    real_dec_cost += real_dec
                    real_rel_cost += real_rel
                    pred_cost += pred_t
                    pred_init_cost += pred_init
                    pred_skip_cost += pred_skip
                    pred_dec_cost += pred_dec
                    pred_rel_cost += pred_rel
                    err, init_err, skip_err, dec_err, rel_err = get_err(
                        pred=[pred_t, pred_init, pred_skip, pred_dec, pred_rel],
                        real=[real_t, real_init, real_skip, real_dec, real_rel]
                    )
                    data = {
                        'n_skip': str(i),
                        'tile pos': pos,
                        'resolution': f'{int(tile.w)}x{int(tile.h)}',
                        'real cost': f'{real_t:.2f}ms, {real_init:.2f}ms, {real_skip:.2f}ms, {real_dec:.2f}ms, {real_rel:.2f}ms',
                        'pred cost': f'{pred_t:.2f}ms, {pred_init:.2f}ms, {pred_skip:.2f}ms, {pred_dec:.2f}ms, {pred_rel:.2f}ms',
                        'error': f'{err:.2f}%, {init_err:.2f}%, {skip_err:.2f}%, {dec_err:.2f}%, {rel_err:.2f}%'
                    }
                    data = [(str(k), str(v)) for k, v in data.items()]
                    print(tabulate(data))
                break
    err, init_err, skip_err, dec_err, rel_err = get_err(
        pred=[pred_cost, pred_init_cost, pred_skip_cost, pred_dec_cost, pred_rel_cost],
        real=[real_cost, real_init_cost, real_skip_cost, real_dec_cost, real_rel_cost]
    )
    data = {
        'real cost': f'{real_cost:.2f}ms, {real_init_cost:.2f}ms, {real_skip_cost:.2f}ms, {real_dec_cost:.2f}ms, {real_rel_cost:.2f}ms',
        'pred cost': f'{pred_cost:.2f}ms, {pred_init_cost:.2f}ms, {pred_skip_cost:.2f}ms, {pred_dec_cost:.2f}ms, {pred_rel_cost:.2f}ms',
        'error': f'{err:.2f}%, {init_err:.2f}%, {skip_err:.2f}%, {dec_err:.2f}%, {rel_err:.2f}%'
    }
    data = [(str(k), str(v)) for k, v in data.items()]
    print(tabulate(data, headers=['TOTAL', '', '', '', '']))

def get_dnn_info(logs: str, network: str, min: int, step: int, sample_min: int, sample_max: int):
    info = {}
    try:
        log = np.load(os.path.join(logs, f'{network}.npy'))
    except:
        raise RuntimeError('run test/dnn_test.py first to generate logs for dnn')
    xs, ys = res2area(log, min, step, sample_min, sample_max)
    k, b = np.polyfit(xs, ys, 1)
    print(f'{network} R^2={R_square(xs, ys, (k, b))}')
    _, ys = res2area(log, min, step, 0, sample_min / 4)
    a = ys.mean()
    info = {'k': k, 'b': b, 'a': a}
    return info

parser = argparse.ArgumentParser(description='System Init')
parser.add_argument('--input', '-i', type=str, help='path to tiled videos')
parser.add_argument('--logs', '-l', type=str, required=True, help='path to log files')
parser.add_argument('--type', '-t', type=str, required=True, help='codec or dnn')
parser.add_argument('--rounds', '-r', type=int,  default=10)
parser.add_argument('--min', type=int, default=130)
parser.add_argument('--step', type=int, default=64)
parser.add_argument('--sample-min', '-m', type=int, default=0)
parser.add_argument('--sample-max', '-M', type=int, default=np.inf)

if __name__ == '__main__':
    args = parser.parse_args()
    show_args(parser.description, args)
    src = args.input
    logs = args.logs
    log_type = args.type
    rounds = args.rounds
    min = args.min
    step = args.step
    sample_min = args.sample_min
    sample_max = args.sample_max
    if log_type == 'codec':
        decoder_info = {
            'vpf': get_codec_info(logs, 'vpf', min, step, sample_min, sample_max),
        }
        with open('codec.json', 'w') as f:
            json.dump(decoder_info, f, indent=4)

        if src is not None:
            eval_codec(src, 'vpf', rounds)
    elif log_type == 'dnn':
        networks = [x[:-4] for x in os.listdir(logs) if x[-3:] == 'npy']
        assert len(networks) > 0, 'run test/dnn_test.py first to generate logs for dnn'
        dnn_info = {
            network: get_dnn_info(logs, network, min, step, sample_min, sample_max) 
                for network in networks
        }
        with open('dnn.json', 'w') as f:
            json.dump(dnn_info, f, indent=4)