import os, shutil
import sys
sys.path.append('.')
from codec import *
from tile.utils import br_str2int, br_float2str
import time
import argparse
import json
import numpy as np
import torch
from tqdm import tqdm

import cv2
def resize(frames: Segment, size: Size) -> Segment:
    return [cv2.resize(x, size) for x in frames]

def crop(frames: Segment, size: Size) -> Segment:
    h, w = size
    return [x[:h, :w, :].copy() for x in frames]

NUM_FRAMES = 30
def prepare_data(src, dst, config, min_res, max_h, max_w, step):
    dec = CV2Decoder(src)
    src_frames = []
    while len(src_frames) < NUM_FRAMES * 3:
        frame = dec.get_frame()
        if frame is None:
            break
        src_frames.append(frame)
    if os.path.exists(dst):
        rewrite = (input(f'{dst} already exists, rewrite? [Y/n]'))
        if rewrite.lower() not in ['y', '']:
            exit(0)
        shutil.rmtree(dst)
    os.mkdir(dst)
    enc_cfg = PyAVEncoderConfig()
    enc_cfg.fps = 30
    enc_cfg.codec = 'hevc'
    enc_cfg.g = config['gop']
    full_bitrate = br_str2int(config['bitrate'])
    h, w = src_frames[0].shape[:2]
    raw_area = h * w
    total = int(((max_h-min_res)//step+1) * ((max_w-min_res)//step+1))
    with tqdm(total=total) as pbar:
        for h in range(min_res, max_h + 1, step):
            for w in range(min_res, max_w + 1, step):
                area = h * w
                enc_cfg.b = br_float2str(full_bitrate / raw_area * area)
                #frames = crop(src_frames, (h, w))
                frames = resize(src_frames, (h, w))
                video = os.path.join(dst, f'{h}_{w}.mp4')
                enc = PyAVEncoder(video, enc_cfg)
                enc.sync_encode(frames)
                pbar.update(1)
                pbar.set_description(f'encoded {h}x{w} frames (bitrate={enc_cfg.b})')

def eval_decoder(dst, min_res, max_h, max_w, step):
    os.system('sh scripts/clean_cache.sh')
    init_log = np.zeros([(max_h-min_res)//step+1, (max_w-min_res)//step+1])
    dec_log = np.zeros_like(init_log)
    skip_log = np.zeros_like(init_log)
    release_log = np.zeros_like(init_log)
    for i, h in enumerate(range(min_res, max_h + 1, step)):
        for j, w in enumerate(range(min_res, max_w + 1, step)):
            # init a decoder
            t = time.time()
            dec = VPFDecoder(os.path.join(dst, f'{h}_{w}.mp4'), frame_type=torch.Tensor)
            dec.get_frame()
            init_log[i][j] = time.time() - t
            t = time.time()
            for k in range(NUM_FRAMES):
                assert dec.get_frame() is not None, f'failed to decode the {k}-th frame for {h}_{w}.mp4'
            dec_log[i][j] = (time.time() - t) / NUM_FRAMES
            init_log[i][j] -= dec_log[i][j]
            t = time.time()
            for k in range(NUM_FRAMES):
                assert dec.skip_frame(), f'failed to skip the {NUM_FRAMES+k}-th frame for {h}_{w}.mp4'
            skip_log[i][j] = (time.time() - t) / NUM_FRAMES
            t = time.time()
            dec = None
            release_log[i][j] = time.time() - t
            eval_pbar.update(1)
    return init_log, dec_log, skip_log, release_log

parser = argparse.ArgumentParser(description='Codec Test')
parser.add_argument('--input', '-i', type=str, required=True, help='source video (.mp4) or folder')
parser.add_argument('--output', '-o', type=str, default=None, help='path to destination video folder')
parser.add_argument('--log', '-l', type=str, required=True, help='path to log folder')
parser.add_argument('--config', '-c', type=str, required=True, help='path to configuration (.json)')
parser.add_argument('--rounds', '-r', type=int, default=5)
parser.add_argument('--step', '-s', type=int, default=64)
parser.add_argument('--min', '-m', type=int, default=130)
parser.add_argument('--max-h', type=int, default=1080)
parser.add_argument('--max-w', type=int, default=1920)

if __name__ == '__main__':
    args = parser.parse_args()
    src = args.input
    dst = args.output
    log = args.log
    config = json.load(open(args.config, 'r'))
    rounds = args.rounds
    step = args.step
    min_res = args.min
    max_h = args.max_h
    max_w = args.max_w
    if os.path.isdir(src):
        dst = src
    else:
        assert dst is not None, 'set --output/-o to specify a temporary video folder'
        prepare_data(src, dst, config, min_res, max_h, max_w, step)

    # only for cuda initialization
    dec = VPFDecoder(os.path.join(dst, f'{min_res}_{min_res}.mp4'), frame_type=torch.Tensor)
    dec.get_frame()

    # real evaluation
    init_log = np.zeros([(max_h-min_res)//step+1, (max_w-min_res)//step+1])
    dec_log = np.zeros_like(init_log)
    skip_log = np.zeros_like(init_log)
    release_log = np.zeros_like(init_log)
    total = int(((max_h-min_res)//step+1) * ((max_w-min_res)//step+1)) * rounds
    with tqdm(total=total, desc='eval') as eval_pbar:
        for i in range(rounds):
            init, dec, skip, release = eval_decoder(dst, min_res, max_h, max_w, step)
            init_log += init
            dec_log += dec
            skip_log += skip
            release_log += release
    np.save(os.path.join(log, 'vpf_init.npy'), init_log / rounds)
    np.save(os.path.join(log, 'vpf_dec.npy'), dec_log / rounds)
    np.save(os.path.join(log, 'vpf_skip.npy'), skip_log / rounds)
    np.save(os.path.join(log, 'vpf_rel.npy'), release_log / rounds)