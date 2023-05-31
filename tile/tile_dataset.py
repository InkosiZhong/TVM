from torch.utils.data import Dataset
import torch
from torchvision.transforms.functional import to_tensor
from typing import Tuple, List, Callable, Union
import sys, cv2
import time
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
sys.path.append('.')
from codec import Size, Frame, BaseDecoder, CV2Decoder, VPFDecoder
from tile import AsyncTileDecoder, TilePos, FrameRange, Rect, Layout
try:
    from utils import str2tuple, tuple2str, binary_search, hit_in, get_tile_path
    from utils import BaseBuffer, LRUBuffer
    from metadata_proxy import MetadataProxy
except:
    from .utils import str2tuple, tuple2str, binary_search, hit_in, get_tile_path
    from .utils import BaseBuffer, LRUBuffer
    from .metadata_proxy import MetadataProxy

TileIdx = Tuple[int, TilePos]
class TileRandomSelectDataset:
    '''
    This dataset enables random selection
    '''
    def __init__(self, 
        root: str, 
        dec_type: BaseDecoder.__class__,  # set None as auto
        target_size: Size, 
        transform_fn: Callable[[torch.Tensor], torch.Tensor]=lambda x, y: x,
        fast_seek: bool=False,      # avoid seek from the I-frame
        codec_cache_size: int=0,    # determine how many decoders can be cached to avoid re-create
        frame_cache_size: int=0,    # determine how many frames can be cached to avoid re-decode
        cache_update_alg: BaseBuffer.__class__=LRUBuffer,
        cache_unhit: bool=False,    # determine whether cache tmp
        max_workers: int=8
    ):
        self.root = root
        self.dec_type = dec_type
        self.get_metadata()
        self.list_of_idxs = []
        self.target_size = target_size
        self.transform_fn = transform_fn
        self.enable_fast_seek = fast_seek
        self.codec_cache_size = codec_cache_size
        self.frame_cache_size = frame_cache_size
        self.cache_update_alg = cache_update_alg
        self.init_cache()
        self.cache_unhit = cache_unhit
        self.enable_cache = True
        self.pool = ThreadPoolExecutor(max_workers=max_workers)
        self.dec_time = 0
        self.discard_t = 0

    def get_metadata(self):
        self.proxy = MetadataProxy()
        self.proxy.read(self.root)
        self.segments = self.proxy.get_segments()
        self.gop_size = self.proxy.get_gop()

    def init_cache(self):
        self.codec_cache = self.cache_update_alg(self.codec_cache_size, mutex=True)
        self.frame_cache = self.cache_update_alg(self.frame_cache_size, mutex=False)

    def init_decoder(self, t: int, dec_pos: TilePos) -> BaseDecoder:
        dec_range = self.get_decode_range(t)
        id = (dec_range, dec_pos)
        path = get_tile_path(self.root, dec_range, dec_pos)
        dec = self.codec_cache[id]
        if dec is None:
            if self.dec_type is CV2Decoder:
                dec = CV2Decoder(path, id)
            elif self.dec_type is VPFDecoder:
                dec = VPFDecoder(path, id, target_size=self.target_size, frame_type=torch.Tensor)
        return dec

    def get_decode_range(self, t: int) -> FrameRange:
        idx = binary_search(self.segments, t, hit_in)
        return self.segments[idx]

    def get_frame_num(self) -> int:
        return self.proxy.count_frames()
    
    def __len__(self):
        return self.proxy.count_roi_tiles()
    
    def fast_seek(self, dec: BaseDecoder, offset: int, pos: TilePos, tile: Rect, sub_layout: Layout) -> bool:
        try:
            for tt in range(offset):
                if self.enable_cache and self.cache_unhit:
                    assert self.dec_type is VPFDecoder, 'only VPFDecoder is enabled yet'
                    surf = dec.get_raw_surface()    # raw surface
                    idx = (tt, pos)
                    if idx in self.frame_cache:
                        self.frame_cache.update(idx)
                    else:
                        self.frame_cache[idx] = (surf, tile, sub_layout, dec.surf_cvt)
                    assert surf is not None, f'error when seeking for {tt}'
                else:
                    assert dec.skip_frame(), f'error when seeking for {tt}'
            return True
        except:
            return False

    def get_tile(self, idx: TileIdx):
        t, pos = idx
        sub_layout = None
        ret = self.frame_cache[idx]
        err = False
        if ret is None:
            dec = self.init_decoder(t, pos)
            start_pos = t - dec.id[0][0]
            offset = start_pos - dec.curr_idx()
            curr_gop = dec.curr_idx() // self.gop_size
            target_gop = start_pos // self.gop_size
            tile = self.get_tile_info(dec.id)
            sub_layout = self.get_sub_layout(dec.id)
            if not self.enable_fast_seek or offset < 0 or curr_gop != target_gop:
                assert dec.set_start_pos(start_pos), f'error when seeking for {t}'
            else: # this will avoid re-seak from the I-frame
                err = not self.fast_seek(dec, offset, pos, tile, sub_layout)
            if not err:
                try:
                    frame = dec.get_frame()
                except:
                    err = True
            if err:
                print(f'failed to decode frame {idx}, recreate decoder and return zeros')
                frame = dec.empty_frame()
                self.codec_cache[dec.id] = None # manual discard
            if self.dec_type is CV2Decoder and self.target_size is not None:
                frame = cv2.resize(frame, self.target_size)
            if self.enable_cache:
                self.codec_cache[dec.id] = dec
        else:
            frame, tile, sub_layout, surf_cvt = ret
            if surf_cvt is not None:
                frame = surf_cvt(frame)
        if not isinstance(frame, torch.Tensor):
            frame = to_tensor(frame).cuda()
        if self.enable_cache:
            self.frame_cache[idx] = (frame, tile, sub_layout, None) # fully decoded frame
        frame = self.transform_fn(frame, tile)
        return frame, tile, sub_layout

    def get_tile_info(self, id):
        enc_range = id[0]
        tile_pos = id[1]
        return self.proxy.get_tile_rect(enc_range, tile_pos)
    
    def get_sub_layout(self, id):
        enc_range = id[0]
        tile_pos = id[1]
        return self.proxy.get_sub_layout(enc_range, tile_pos)

    def get_tiles(self, idxs: List[TileIdx]):
        tasks = []
        for idx in idxs:
            task = self.pool.submit(self.get_tile, idx)
            tasks.append(task)
        wait(tasks, return_when=ALL_COMPLETED)
        return [t.result() for t in tasks]
    
    def get_tiles_in_frame(self, t: int):
        dec_range = self.get_decode_range(t)
        all_pos = self.proxy.get_tile_pos(dec_range)
        idxs = [(t, pos) for pos in all_pos]
        return self.get_tiles(idxs)
    
    def get_decode_time(self) -> float:
        '''
        this function is used to return the time it cost for decoding
        '''
        return self.dec_time
    
    def __getitem__(self, idx: Union[int, TileIdx, List[TileIdx]]):
        start = time.time()
        if isinstance(idx, int):
            ret = self.get_tiles_in_frame(idx)
        elif isinstance(idx, tuple):
            ret = self.get_tile(idx)
        elif isinstance(idx, list):
            ret = self.get_tiles(idx)
        else:
            raise RuntimeError
        self.dec_time += time.time() - start
        return ret

class TileFullScanDataset(Dataset):
    '''
    a pytorch dataset
    '''
    def __init__(self, 
        root: str, 
        dec_type: BaseDecoder.__class__, 
        target_size: Size, max_workers: int=4, 
        I_frame_only: bool=False,
        info_filter: Callable[[any], bool]=None, # filter tiles by additional info
        transform_fn: Callable[[torch.Tensor], torch.Tensor]=lambda x, y: x
    ) -> None:
        self.root = root
        self.get_metadata()
        self.length = 0
        self.async_dec = AsyncTileDecoder(root, dec_type, target_size, torch.Tensor, max_workers, 16 * 30)
        self.cnt = 0
        self.transform_fn = transform_fn

        for enc_range in self.segments:
            s, e = enc_range
            for tile_pos in self.proxy.get_tile_pos(enc_range):
                valid = True
                if info_filter is not None:
                    info = self.proxy.get_tile_info(enc_range, tile_pos)
                    valid = info_filter(info)
                if valid:
                    self.async_dec.submit(
                        enc_range, tile_pos, 
                        [(s, s+1)] if I_frame_only else [(s, e)]
                    )
                    self.length += 1 if I_frame_only else e-s

    def get_metadata(self):
        self.proxy = MetadataProxy()
        self.proxy.read(self.root)
        self.segments = self.proxy.get_segments()

    def __len__(self):
        return self.length

    def get_frame_num(self) -> int:
        return self.proxy.count_frames()

    def get_decode_range(self, t: int) -> FrameRange:
        idx = binary_search(self.segments, t, hit_in)
        return self.segments[idx]

    def __getitem__(self, index) -> Tuple[int, TilePos, torch.Tensor]:
        '''
        return Frame Number, Tile Position and Frame
        '''
        id, idx, frame = self.async_dec.get_item()
        t = id[0] + idx
        pos = id[1:]
        enc_range = self.get_decode_range(t)
        tile = self.proxy.get_tile_rect(enc_range, pos)
        frame = self.transform_fn(frame, tile)
        return (t, pos, frame)
        
if __name__ == '__main__':
    from codec import *
    dataset = TileRandomSelectDataset(
        '../datasets/jackson-town-square/2017-12-17-tile',
        VPFDecoder,
        target_size=None,
        fast_seek=True,
        codec_cache_size=10,
        frame_cache_size=0
    )
    for _ in range(100):
        x = dataset[[(210 + i, (1, 1)) for i in range(8)]]
        print(len(x))