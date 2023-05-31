import sys, os
sys.path.append('.')
import torch
from tqdm import tqdm
from codec import *
from scripts.script_utils import SegDecoder, show_args
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, ALL_COMPLETED
import argparse

def init_decoder(src):
    if os.path.isdir(src):
        dec = SegDecoder(src, dec_type=VPFDecoder, frame_type=torch.Tensor)
    else:
        dec = VPFDecoder(src, frame_type=torch.Tensor)
    return dec

def eval_segment(seg: str, nf: int):
    dec = init_decoder(seg)
    name = seg.strip().split('/')[-1]
    for i in range(nf):
        try:
            x = dec.get_frame()
            if x is None:
                print(f'wrong nf {i}<{nf} ({name})')
                return
        except:
            print(f'crash at {i} ({name})')
            return i, name
    return None

parser = argparse.ArgumentParser(description='Find Crash')
parser.add_argument('--input', '-i', type=str, required=True, help='source video (.mp4) or folder')
parser.add_argument('--untiled', '-u', type=str, help='path to untiled video')

if __name__ == '__main__':
    args = parser.parse_args()
    show_args(parser.description, args)
    src = args.input
    untiled = args.untiled
    tmp = init_decoder(src)
    nf_list = tmp.nf_list
    segments = tmp.segments
    crash_frames = []
    crash_segs = set()
    if not untiled:
        pool = ThreadPoolExecutor(max_workers=8)
        tasks = []
        for seg, nf in tqdm(zip(segments, nf_list), total=len(nf_list)):
            dec = init_decoder(seg)
            name = seg.strip().split('/')[-1]
            task = pool.submit(eval_segment, seg, nf)
            tasks.append(task)
            if len(tasks) == 8:
                wait(tasks, return_when=FIRST_COMPLETED)
                unfinish_tasks = []
                for task in tasks:
                    if task.done():
                        x = task.result()
                        if x:
                            crash_frames.append(x[0])
                            crash_segs.add(x[1])
                    else:
                        unfinish_tasks.append(task)
                tasks = unfinish_tasks
        wait(tasks, return_when=ALL_COMPLETED)
        for task in tasks:
            x = task.result()
            if x:
                crash_frames.append(x[0])
                crash_segs.add(x[1])
    else:
        seg_idx, cnt = 0, 0
        dec = init_decoder(untiled)
        assert int(dec.nf) == int(tmp.nf), f'frame number not match, {int(dec.nf)} ({src}), {int(tmp.nf)} ({untiled})'
        for i in tqdm(range(int(dec.nf))):
            try:
                x = dec.get_frame()
                if x is None:
                    break
            except:
                seg = segments[seg_idx].strip().split('/')[-1]
                start = sum(nf_list[:seg_idx])
                end = start + nf_list[seg_idx]
                print(f'crash at {i} ({seg}, {start}-{end})')
                crash_frames.append(i)
                crash_segs.add((seg, start, end))
                dec = init_decoder(untiled)
                dec.set_start_pos(i + 1)
            cnt += 1
            if cnt >= nf_list[seg_idx]:
                seg_idx += 1
                cnt = 0
    print(f'all crash frames: {crash_frames}')
    print(f'all crash segments: {crash_segs}')