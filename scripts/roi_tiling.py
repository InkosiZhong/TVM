import sys, os, shutil
sys.path.append('.')
import json
from tqdm import tqdm
from tabulate import tabulate
from tile import *
from codec import *
from scripts.script_utils import SegDecoder, show_args, show_config
from tiling_index import *
import time
import argparse

def create_encoder(enc_type, roi_only, max_workers=8):
    if enc_type == PyAVEncoder:
        enc_cfg = PyAVEncoderConfig()
        enc_cfg.codec = 'hevc'
        enc_cfg.g = gop
        enc_cfg.b = cfg['bitrate']
    elif enc_type == VPFEncoder:
        enc_cfg = VPFEncoderConfig()
        enc_cfg.gop = gop
        enc_cfg.bitrate = cfg['bitrate']
    else:
        raise RuntimeError(f'unsupported encoder type {enc_type}')
    # enable RA mode
    #enc_cfg.preset = 'ultrafast'
    #enc_cfg.preset = None
    #enc_cfg.bf = None
    if enc_type == PyAVEncoder:
        enc = TileEncoder(dst, PyAVEncoder, enc_cfg, roi_only=roi_only, max_workers=max_workers)
    elif enc_type == VPFEncoder:
        enc = TileEncoder(dst, VPFEncoder, enc_cfg, roi_only=roi_only, max_workers=max_workers)
    return enc

parser = argparse.ArgumentParser(description='ROI Tiling')
parser.add_argument('--input', '-i', type=str, required=True, help='source video (.mp4) or folder')
parser.add_argument('--output', '-o', type=str, required=True, help='path to destination video')
parser.add_argument('--config', '-c', type=str, required=True, help='path to configuration (.json)')
parser.add_argument('--train-cache', '-t', type=str, default=None, help='path to train cache')
parser.add_argument('--limit', '-n', type=int, default=None, help='number of frames')
parser.add_argument('--save-all', action='store_true')

if __name__ == '__main__':
    args = parser.parse_args()
    show_args(parser.description, args)
    src = args.input
    dst = args.output
    cfg = json.load(open(args.config, 'r'))
    show_config(args.config, cfg)
    train_cache = args.train_cache
    if train_cache:
        codec_info = json.load(open('codec.json', 'r'))['vpf']
    else:
        codec_info = None
    limit = args.limit
    roi_only = not args.save_all
    
    if train_cache is not None and train_cache != 'merge-all':
        idx_cfg = TilingConfig()
        idx_cfg.cache_root = train_cache
        idx_cfg.do_mining = False
        idx_cfg.do_training = False
        index = TilingIndex(idx_cfg, "", cfg['classes'], 0.5)
        index.init()

    valid_area = str2rect(cfg['valid_area'])
    ban_area = [str2rect(x) for x in cfg['ban_area']]
    if os.path.isdir(src):
        dec = SegDecoder(src, dec_type=CV2Decoder)
    else:
        dec = CV2Decoder(src)
    skip_area = ban_area #+ exclude(Rect(0, 0, w=dec.width, h=dec.height), valid_area)
    f = ROIFetcher(
        'ViBe', 
        filter_list=[
            create_area_filter(skip_area), 
            create_size_filter([*cfg['roi_min_size']])
        ],
        #scale_factor=0.125,
        inflater_func=create_naive_inflater(min_size=cfg['inflater_size']), 
        blur_kernel=cfg['blur_kernel']
    )
    lay_gen = LayoutGenerator((dec.height, dec.width), condition_list=[
        create_ctr_rel_close_condition(cfg['ctr_dist_k']),
        intersect,
    ], min_size=(cfg['tile_min_size'], cfg['tile_min_size'])) # min h and w for VPF(HEVC)
    gop = cfg['gop']
    seg = cfg['segment']
    sim_threshold = cfg['sim_threshold']
    seg_roi = []
    frames = []
    cost_func = create_codec_cost_func(codec_info, gop)
    if os.path.exists(dst):
        shutil.rmtree(dst)
        os.mkdir(dst)
    enc = create_encoder(PyAVEncoder, roi_only, 8)
    nf = int(dec.nf)
    nf = min(limit, nf) if limit is not None else nf
    t = time.time()
    for i in tqdm(range(nf), f'{src.strip("/").split("/")[-1]}->{dst.strip("/").split("/")[-1]}'):
        frame = dec.get_frame()
        if frame is None:
            break
        if i > 0 and i % seg == 0:
            layout, seg_roi = lay_gen.get_layout(seg_roi)
            if train_cache is not None:
                if train_cache == 'merge-all':
                    layout = merge_all(layout)
                else:
                    adj_layout = semantic_tiling(index, cost_func, frames, layout, seg_roi, sim_threshold)
                    layout = hierarchical_layout(layout, adj_layout)
            enc.encode(frames, layout, new_tile=True)
            enc.wait_for_idle()
            seg_roi = []
            frames = []
        frames.append(frame)
        roi = f.fetch(frame)
        valid_roi = []
        for r in roi:
            inter = intersect_rect(r, valid_area)
            if inter is not None and inter.area() > 0.5 * r.area():
                valid_roi.append(r)
        seg_roi.append(valid_roi)
    layout, seg_roi = lay_gen.get_layout(seg_roi)
    if train_cache is not None:
        if train_cache == 'merge-all':
            layout = merge_all(layout)
        else:
            adj_layout = semantic_tiling(index, cost_func, frames, layout, seg_roi, sim_threshold)
            layout = hierarchical_layout(layout, adj_layout)
    enc.encode(frames, layout, new_tile=True)
    enc.flush()
    enc.join()
    print(f'tiling {nf} frames in {nf / (time.time() - t):.2f}fps')