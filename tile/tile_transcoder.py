from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, ALL_COMPLETED, FIRST_COMPLETED
import cv2, torch
from torchvision.transforms.functional import to_tensor
from queue import Queue
import time
import numpy as np
from typing import List, Tuple, Union, Optional
from queue import Queue
import os, sys, copy
sys.path.append('.')
from codec import BaseEncoder, BaseDecoder, BaseEncoderConfig, PyAVEncoderConfig, VPFEncoderConfig
from codec import Segment, Frame, Size, CV2Decoder, VPFDecoder
from tile import Layout, Rect
from .utils import get_tile_path, br_str2int, br_float2str
from .metadata_proxy import MetadataProxy

FrameRange = Tuple[int, int]    # (start, end)
TilePos = Tuple[int, int]       # (row, col)

def pack(encoder_type: BaseEncoder.__class__, name: str, cfg: BaseEncoderConfig, tiles: Segment, **encoder_args):
    '''
    this function is a package for ThreadPoolExecutor to avoid memory overflow
    > UserWarning: stream in out-of-thread context could not be cleaned up
    '''
    encoder: BaseEncoder = encoder_type(name, cfg, **encoder_args)
    encoder.sync_encode(tiles)

class TileEncoder:
    def __init__(self, 
        base_path: str,
        encoder_type: BaseEncoder.__class__,    # class name of the encoder
        cfg: BaseEncoderConfig,                 # coresponding configuration
        roi_only: bool = True,                  # only save tiles with roi
        max_workers: int = 8,
        **encoder_args,                         # other arguments of the encoder, such as gpu_id
    ) -> None:
        '''
        This class will encode a Segment into Tiles
        '''
        self.base_path = base_path
        self.encoder_type = encoder_type
        self.cfg = cfg
        self.roi_only = roi_only
        self.encoder_args = encoder_args
        self.max_workers = max_workers
        self.pool = ThreadPoolExecutor(max_workers=max_workers)
        #self.pool = ProcessPoolExecutor(max_workers=max_workers)
        self.frame_cnt = 0
        self.tile_cnt = 0
        self.tile_roi_cnt = 0
        self.frame_buffer = []                  # buffer frames if layout don't change
        self.curr_layout = []
        self.tasks = []
        self.proxy = MetadataProxy()
        self.w = None
        self.h = None
        self.area = None
        if isinstance(cfg, PyAVEncoderConfig):
            self.gop = cfg.g
            self.full_bitrate = br_str2int(cfg.b)
        elif isinstance(cfg, VPFEncoderConfig):
            self.gop = cfg.gop
            self.full_bitrate = br_str2int(cfg.bitrate)
        else:
            raise RuntimeError(f'Unsupported configure type {type(cfg)}')
        self.fps = cfg.fps

    def encode(self, frames: Segment, layout: Layout, new_tile: bool=False):
        if self.w is None or self.h is None:
            self.h, self.w = frames[0].shape[:2]
            self.area = self.h * self.w
        if new_tile or layout != self.curr_layout: # encode if layout changes
            num_frame = len(self.frame_buffer)
            if num_frame > 0:
                enc_range = (self.frame_cnt, self.frame_cnt + num_frame)
                self.proxy.record_layout(enc_range, self.curr_layout)
                self.append_encoder(num_frame)
                self.frame_cnt += num_frame
            self.frame_buffer = frames
            self.curr_layout = layout
        else:
            self.frame_buffer += frames

    def flush(self):
        num_frame = len(self.frame_buffer)
        if num_frame > 0:
            enc_range = (self.frame_cnt, self.frame_cnt + num_frame)
            self.proxy.record_layout(enc_range, self.curr_layout)
            self.append_encoder(num_frame)
        self.frame_cnt += num_frame
        self.frame_buffer = []
        self.curr_layout = []
        self.proxy.record_summarize({
            "width": self.w,
            "height": self.h,
            "gop": self.gop
        })
        self.proxy.write(root=self.base_path)
        self.frame_cnt = 0

    def set_start_pos(self, start_pos: int):
        self.frame_cnt = start_pos

    def append_encoder(self, num_frame):
        N, M = len(self.curr_layout), len(self.curr_layout[0])
        for i in range(N):
            for j in range(M):
                rect, has_roi = self.curr_layout[i][j][:2]
                if self.roi_only and not has_roi:
                    continue
                name = self.get_tile_name(self.base_path, num_frame, (i, j))
                tiles = self.get_tile(self.frame_buffer, rect)
                cfg = self.get_config(rect)
                task = self.pool.submit(pack, self.encoder_type, name, cfg, tiles, **self.encoder_args)
                self.tasks.append(task)

    def get_config(self, tile: Rect) -> BaseEncoderConfig:
        ratio = tile.area() / self.area
        bitrate = self.full_bitrate * ratio
        cfg = copy.copy(self.cfg)
        if isinstance(cfg, PyAVEncoderConfig):
            cfg.b = br_float2str(bitrate)
        elif isinstance(cfg, VPFEncoderConfig):
            cfg.bitrate = br_float2str(bitrate)
        return cfg

    def get_tile_name(self, base_path: str, num_frame: int, tile_pos: TilePos) -> str:
        row, col = tile_pos
        root = os.path.join(base_path, f'{self.frame_cnt}_{self.frame_cnt+num_frame}')
        if not os.path.exists(root):
            os.makedirs(root)
        return os.path.join(root, f'{row}_{col}.mp4')

    def get_tile(self, frames: Segment, rect: Rect) -> Segment:
        x1, y1, x2, y2 = rect.xyxy()
        tile = [frame[int(y1):int(y2), int(x1):int(x2), :].copy() for frame in frames]
        return tile

    def join(self):
        wait(self.tasks, return_when=ALL_COMPLETED)
        self.tasks = [x for x in self.tasks if not x.done()]

    def wait_for_idle(self):
        while len(self.tasks) >= self.max_workers:
            wait(self.tasks, return_when=FIRST_COMPLETED)
            self.tasks = [x for x in self.tasks if not x.done()]

def tile_metadata_helper(root: str) -> TilePos:
    tiles = os.listdir(root)
    cols = len([None for t in tiles if t.split('_')[0] == '0'])
    rows = len(tiles) // cols
    return rows, cols

class AsyncTileDecoder:
    '''
    Async & Out of order
    '''
    def __init__(self, 
        root: str, 
        dec_type: BaseDecoder.__class__, 
        target_size: Size, 
        frame_type: type=Frame, 
        max_workers: int=4,
        queue_size: int = 16 * 60 # avoiding memory overflow
    ) -> None:
        self.root = root
        self.dec_type = dec_type
        self.target_size = target_size
        self.frame_type = frame_type
        self.queue = Queue(queue_size)
        self.pool = ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, enc_range: FrameRange, tile_pos: TilePos, dec_ranges: List[FrameRange]):
        '''
        enc_range: specify the gop, [s, e)
        tile_pos: specify the position
        dec_ranges: only decode valid parts, [[s, e)]
        '''
        start_pos, limit, skip_list = self.get_decode_param(enc_range, dec_ranges)
        path = get_tile_path(self.root, enc_range, tile_pos)
        id = (enc_range[0], *tile_pos)  # (start, row, col)
        self.pool.submit(self.start, path, id, start_pos, limit, skip_list)

    def init_decoder(self, path, id):
        if self.dec_type is VPFDecoder:
            dec = VPFDecoder(path, id, self.queue, target_size=self.target_size, frame_type=self.frame_type)
        elif self.dec_type is CV2Decoder:
            dec = CV2Decoder(path, id, self.queue)
        else:
            raise TypeError(f'invalid decoder type {self.dec_type}')
        return dec

    def start(self, path: str, id: tuple, start_pos: int, limit: Optional[int], skip_list: List[int]):
        dec = self.init_decoder(path, id)
        if start_pos > 0:
            dec.set_start_pos(start_pos)
        try:
            dec.decode_all(limit, skip_list)
        except:
            print(f'error when decode {path}({dec.curr_idx()}), re-create decoder and return zeros')
            idx = dec.curr_idx()
            self.queue.put((dec.id, idx, dec.empty_frame()))
            limit = limit - dec.offset() - 1 if limit is not None else None
            self.pool.submit(self.start, path, id, idx+1, limit, skip_list)

    def get_decode_param(self, enc_range: FrameRange, dec_ranges: List[FrameRange]) -> Tuple[int, int, List[int]]:
        '''
        input enc_range and dec_ranges information
        return (start_pos, limit, skip_list)
        '''
        s, e = enc_range
        skip_list = [True for _ in range(e-s)]
        for dec_range in dec_ranges:
            dec_s, dec_e = dec_range
            assert s <= e and dec_s <= dec_e, f'invalid range {dec_range} in {enc_range}'
            if dec_s < s:
                dec_s = s
            if dec_e > e:
                dec_e = e
            for i in range(dec_s, dec_e):
                skip_list[i-s] = False
        start_pos = 0
        limit = None
        for i, skip in enumerate(skip_list):
            if not skip and start_pos == 0:
                start_pos = i
                break
        for i, skip in enumerate(reversed(skip_list)):
            if not skip and limit is None:
                limit = e - s - i - start_pos
                break
        skip_list = [i for i, skip in enumerate(skip_list) if skip and i >= start_pos and i - start_pos < limit]
        return start_pos, limit, skip_list

    def get_item(self) -> Tuple[TilePos, int, Union[Frame, torch.Tensor]]:
        '''
        return Frame|torch.Tensor, Tile Position and Frame Number
        this function will block when there is no more frames to decode, 
        control the times of call by the main program
        '''
        id, idx, frame = self.queue.get()
        if self.dec_type is CV2Decoder:
            if self.target_size is not None:
                frame = cv2.resize(frame, self.target_size)
            if self.frame_type is torch.Tensor:
                frame = to_tensor(frame)
        return id, idx, frame

Grid = List[List]
class TileDecoder:
    def __init__(self,
        base_path: str,
        dec_type: BaseDecoder.__class__,
        max_workers: int = 8,
    ) -> None:
        '''
        This class will recovery a Segment from Tiles
        '''
        self.base_path = base_path
        self.root_list = sorted([x for x in os.listdir(self.base_path) if '.' not in x], 
            key=lambda x:int(x.split('_')[0]))
        self.async_dec = AsyncTileDecoder(base_path, dec_type, None, Frame, max_workers)
        self.tile_grid: List[Grid] = []     # convert into frame when a grid is full
        self.cnt = 0

    def set_decode_range(self, dec_range: FrameRange=(0, 9999)):
        '''
        |<--start_pos-->|<--limit-->|     |
        s            valid_s     valid_e  e
        '''
        dec_range = self.get_valid_range(self.root_list, dec_range)
        self.valid_s, self.valid_e = dec_range
        for root in self.root_list:
            s, e = self.parse_range(root)
            if self.valid_s >= e: # skip this segment
                continue
            root = os.path.join(self.base_path, root)
            self.append_tile_grid(root, (s, e), dec_range)
            self.append_decoder(root, (s, e), dec_range)
            if self.valid_e <= e: # skip post segments
                break

    def get_valid_range(self, root_list:List[str], range: FrameRange) -> FrameRange:
        i_start, i_end = range
        assert i_end > i_start, f'i_end <= i_start'
        s, _ = self.parse_range(root_list[0])
        _, e = self.parse_range(root_list[-1])
        assert i_start < e, f'invalid i_start {i_start}>={e}'
        i_start, i_end = max(i_start, s), min(i_end, e)
        print(f'decode frames {i_start}:{i_end}')
        return i_start, i_end

    def parse_range(self, root: str) -> FrameRange:
        s, e = root.split('_')
        return int(s), int(e)

    def append_tile_grid(self, root: str, enc_range: FrameRange, dec_range: FrameRange):
        # TODO: use database to record
        rows, cols = tile_metadata_helper(root)
        enc_s, enc_e = enc_range
        dec_s, dec_e = dec_range
        s, e = max(enc_s, dec_s), min(enc_e, dec_e)
        num_frames = max(e-s, 0)
        for _ in range(num_frames):
            self.tile_grid.append([[
                None for _ in range(cols)
            ] for _ in range(rows)])

    def append_decoder(self, root: str, enc_range: FrameRange, dec_range: FrameRange):
        rows, cols = tile_metadata_helper(root)
        for i in range(rows):
            for j in range(cols):
                self.async_dec.submit(enc_range, (i, j), [dec_range])
        
    def get_frame(self) -> Frame:
        '''
        decode and merge into one frame
        '''
        if self.done():
            return None
        # collect tiles from queue
        while not self.collect_complete(self.tile_grid[0]):
            id, idx, frame = self.async_dec.get_item()
            gop_idx, row, col = id
            idx += gop_idx-self.valid_s
            self.tile_grid[idx-self.cnt][row][col] = frame
        # merge into one frame
        grid = self.tile_grid.pop(0)
        for i in range(len(grid)):
            grid[i] = np.concatenate(grid[i], axis=1)
        frame = np.concatenate(grid)
        self.cnt += 1
        return frame

    def collect_complete(self, grid: Grid) -> bool:
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] is None:
                    return False
        return True

    def done(self):
        return len(self.tile_grid) == 0

    def get_tile_name(self, root: str, tile_pos: TilePos) -> str:
        row, col = tile_pos
        return os.path.join(root, f'{row}_{col}.mp4')

if __name__ == '__main__':
    from filter import create_area_filter, create_size_filter
    from roi_fetcher import ROIFetcher
    from codec import *
    from condition import create_edge_abs_close_condition, intersect
    from layout_generator import LayoutGenerator
    import shutil
    if os.path.exists('out'):
        shutil.rmtree('out')
        os.mkdir('out')
    #skip_area = [Rect(40, 15, w=450, h=20), Rect(120, 475, w=250, h=50)]
    skip_area = [Rect(0, 0, w=1920, h=400), Rect(0, 1060, w=1920, h=20)]
    f = ROIFetcher('ViBe', filter_list=[
        create_area_filter(skip_area), 
        create_size_filter([5, 5])
    ])
    dec = CV2Decoder('./data/2017-12-14.mp4')
    cfg = VPFEncoderConfig()
    #cfg.codec = 'h264'
    enc = TileEncoder('out', VPFEncoder, cfg, max_workers=4)
    lay_gen = LayoutGenerator((dec.height, dec.width), condition_list=[
        create_edge_abs_close_condition(0.02 * dec.height),
        intersect,
    ], min_size=(130, 130)) # min h and w for VPF(HEVC)
    gop_roi = []
    frames = []
    cnt = 0
    gop = 30
    t = time.time()
    #while cnt < 1500:
    while True:
        frame = dec.get_frame()
        if frame is None:
            break
        if cnt > 0 and cnt % gop == 0:
            layout = lay_gen.get_layout(gop_roi)
            enc.encode(frames, layout)
            gop_roi = []
            frames = []
        frames.append(frame)
        roi = f.fetch(frame)
        for r in roi:
            gop_roi.append(r)
        cnt += 1
    layout = lay_gen.get_layout(gop_roi)
    enc.encode(frames, layout)
    enc.flush()
    enc.join()
    cost = time.time() - t
    print(f'costs {cost:.4f}s to transcode {cnt} frames ({cnt/cost:.2f}fps)')

    '''for i in [1, 2, 4, 8]:
        print(f'{i} workers')
        dec = TileDecoder('out', CV2Decoder , max_workers=i)
        t = time.time()
        dec.set_decode_range((240, 270))
        frames = []
        while not dec.done():
            frame = dec.get_frame()
            print(len(frames), time.time() - t)
            frames.append(frame)
        cost = time.time() - t
        print(f'costs {cost:.4f}s to decode {len(frames)} frames ({len(frames)/cost:.2f}fps)')'''
    '''if os.path.exists('out.mp4'):
        os.remove('out.mp4')
    enc = PyAVEncoder('out.mp4', PyAVEncoderConfig())
    enc.sync_encode(frames)'''

    '''q = Queue()
    pool = ThreadPoolExecutor(8)
    t = time.time()
    tasks = []
    N = 4
    for i in range(N):
        dec = CV2Decoder(f'out/0_30/{i%4}_0.mp4', i, q)
        task = pool.submit(dec.decode_all)
        tasks.append(task)
    print(time.time() - t)
    wait(tasks, return_when=ALL_COMPLETED)
    cost = time.time() - t
    print(q.qsize())
    print(f'{cost:.2f}s ({N*30/4/cost:.2f}fps)')'''

    # test min size for VPFDecoder -> (130, 130)
    '''import cv2
    dec = CV2Decoder('./data/shibuya.mp4')
    frames = [dec.get_frame() for _ in range(30)]
    frames = [cv2.resize(x, (128, 130)) for x in frames]
    if os.path.exists('out.mp4'):
        os.remove('out.mp4')
    enc = PyAVEncoder('out.mp4', PyAVEncoderConfig())
    enc.sync_encode(frames)
    dec = VPFDecoder('out.mp4')
    dec.get_frame()'''