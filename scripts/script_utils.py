import os, sys
sys.path.append('.')
from codec import *
import torch, cv2
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm
#import numpy as np
import json
from tabulate import tabulate

def show_args(desc, args):
    data = vars(args).items()
    headers = [desc.upper(), '']
    print(tabulate(data, headers=headers))

def show_config(desc, cfg):
    headers = [desc.upper(), '']
    data = [(k,v) for k,v in cfg.items() if k not in ['ban_area']]
    print(tabulate(data, headers=headers))

class SegDecoder(torch.utils.data.Dataset):
    def __init__(
        self, src: str, 
        dec_type: BaseDecoder.__class__=CV2Decoder, 
        frame_type: type=Frame, 
        target_size: Size=None) -> None:
        self.src = src
        self.segments = [x for x in os.listdir(src) if x[-3:] == 'mp4']
        self.segments = sorted(self.segments, key=lambda x:int(x[:-4]))
        self.nf_list = self.get_nf_list()
        self.nf = int(sum(self.nf_list))
        self.segments = [os.path.join(src, x) for x in self.segments]
        self.idx = 1
        self.dec_type = dec_type
        self.frame_type = frame_type
        self.target_size = target_size
        self.get_decoder(self.segments[0])
        self.width, self.height = self.dec.width, self.dec.height
        self.frame_idx = 0
        self.cnt = 0
    
    def get_nf_list(self):
        '''if 'nf_list.npy' in os.listdir(self.src):
            print('find nf_list.npy')
            self.nf_list = list(np.load(os.path.join(self.src, 'nf_list.npy')))
        else:
            self.nf_list = []
            for seg in tqdm(self.segments, 'init decoder'):
                self.nf_list.append(CV2Decoder(seg).nf)
            np.save(os.path.join(self.src, 'nf_list.npy'), np.array(self.nf_list))'''
        if 'nf_list.json' in os.listdir(self.src):
            print('find nf_list.json')
            nf_list = json.load(open(os.path.join(self.src, 'nf_list.json')))
        else:
            nf_list = {}
            for seg in tqdm(self.segments, 'init decoder'):
                nf_list[seg] = int(CV2Decoder(os.path.join(self.src, seg)).nf)
            with open(os.path.join(self.src, 'nf_list.json'), 'w') as f:
                json.dump(nf_list, f)
        for seg in self.segments:
            if seg not in nf_list: # not seen segment
                print(f'{seg} is not seen, regenerate nf_list.json')
                os.remove(os.path.join(self.src, 'nf_list.json'))
                return self.get_nf_list()
        pop_keys = []
        for key in nf_list.keys(): # deleted segment
            if key not in self.segments:
                print(f'{key} is not available, pop out')
                pop_keys.append(key)
        for key in pop_keys:
            nf_list.pop(key)
        return list(nf_list.values())

    def get_decoder(self, seg):
        if self.dec_type is CV2Decoder:
            self.dec = CV2Decoder(seg)
        elif self.dec_type is VPFDecoder:
            self.dec = VPFDecoder(seg, target_size=self.target_size, frame_type=self.frame_type)

    def curr_seg(self) -> tuple:
        curr_idx = self.idx-1
        seg = self.segments[curr_idx]
        start = sum(self.nf_list[:curr_idx])
        end = start + self.nf_list[curr_idx]
        return (seg, start, end)

    def get_frame(self) -> Frame:
        frame = self.dec.get_frame()
        if frame is None:
            if self.cnt != self.nf_list[self.idx-1]:
                print(f'wrong nf {self.cnt} != {self.nf_list[self.idx-1]}')
            if self.idx < len(self.segments):
                self.get_decoder(self.segments[self.idx])
                self.idx += 1
                frame = self.dec.get_frame()
                self.cnt = 0
        if frame is not None and self.dec_type is CV2Decoder:
            if self.target_size is not None:
                frame = cv2.resize(frame, self.target_size[::-1])
            if self.frame_type is torch.Tensor:
                frame = to_tensor(frame)
        self.frame_idx += 1
        self.cnt += 1
        return frame

    def set_start_pos(self, pos):
        if pos <= 0:
            return
        cnt = 0
        for i, nf in enumerate(self.nf_list):
            if cnt <= pos < cnt + nf:
                self.idx = i
                self.get_decoder(self.segments[i])
                self.dec.set_start_pos(pos - cnt)
                self.cnt = pos - cnt
                break
            cnt += int(nf)
        self.frame_idx = pos

    def __len__(self):
        return self.nf

    def __getitem__(self, idx):
        return self.frame_idx, self.get_frame()

if __name__ == '__main__':
    dec = SegDecoder('../datasets/test', CV2Decoder)
    from tqdm import tqdm
    for _ in tqdm(range(dec.nf)):
        x = dec.get_frame()
        if x is None:
            break