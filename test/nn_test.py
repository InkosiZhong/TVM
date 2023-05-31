import os
import sys
sys.path.append('.')
from codec import *
import time
import argparse
import json
import numpy as np
import torch
from query.detector import YoloV5
from tqdm import tqdm

NUM_FRAMES = 30

def eval_dnn(model, min_res, max_h, max_w, step):
    dnn_log = np.zeros([(max_h-min_res)//step+1, (max_w-min_res)//step+1])
    for i, h in enumerate(range(min_res, max_h + 1, step)):
        for j, w in enumerate(range(min_res, max_w + 1, step)):
            frame = torch.zeros((1, 3, h, w)).cuda()
            t = time.time()
            with torch.no_grad():
                _ = model(frame)
            dnn_log[i][j] = time.time() - t
            eval_pbar.update(1)
    return dnn_log

parser = argparse.ArgumentParser(description='DNN Test')
parser.add_argument('--log', '-l', type=str, required=True, help='path to log folder')
parser.add_argument('--rounds', '-r', type=int, default=30)
parser.add_argument('--step', '-s', type=int, default=64)
parser.add_argument('--min', '-m', type=int, default=130)
parser.add_argument('--max-h', type=int, default=1080)
parser.add_argument('--max-w', type=int, default=1920)
parser.add_argument('--network', '-n', type=str, default='yolov5m')

if __name__ == '__main__':
    args = parser.parse_args()
    log = args.log
    rounds = args.rounds
    step = args.step
    min_res = args.min
    max_h = args.max_h
    max_w = args.max_w
    network = args.network

    model = YoloV5(model_capacity=network, scale_factor=1).eval().cuda()
    dnn_log = np.zeros([(max_h-min_res)//step+1, (max_w-min_res)//step+1])
    total = int(((max_h-min_res)//step+1) * ((max_w-min_res)//step+1)) * rounds
    with tqdm(total=total, desc='eval') as eval_pbar:
        for i in range(rounds):
            dnn = eval_dnn(model, min_res, max_h, max_w, step)
            dnn_log += dnn
    np.save(os.path.join(log, f'{network}.npy'), dnn_log / rounds)