from threading import Thread
import numpy as np
import torch
import os
import time
from queue import Queue
from typing import List, Tuple, Union, Optional
try:
    from config import BaseEncoderConfig
except:
    from .config import BaseEncoderConfig
Frame = np.array
Segment = List[Frame]
Size = Tuple[int, int] # (height, width)

T_SLEEP = 1e-3 # s
QUEUE_SIZE_LIMIT = 1000
class BaseEncoder(Thread):
    def __init__(self, file_name: str, cfg: BaseEncoderConfig):
        Thread.__init__(self)
        assert not os.path.exists(file_name), f'{file_name} already exists'
        self.file_name = file_name
        self.frames = Queue(maxsize=QUEUE_SIZE_LIMIT)    # FIFO
        self.cfg = cfg
        self.enc_frame = None
        self.ready = False
        self.finish_sign = False
        self.finished = False

    def dynamic_init(self, frame: Frame):
        '''
        [optional] do some frame aware initialization
        '''
        self.ready = True

    def encode(self, frames: Segment):
        '''
        asynchronous encoding
        encode single frame by [frame]
        '''
        if not self.ready and len(frames) > 0:
            self.dynamic_init(frames[0])
        for frame in frames:
            self.frames.put(frame)      # append at last

    def finish(self):
        '''
        encoder will not exit at once, 
        until all frames are written and all resources are released
        '''
        self.finish_sign = True

    def done(self) -> bool:
        '''
        return if the encoder written all frames and release all resoureces
        '''
        return self.finished

    def sync_encode(self, frames: Segment):
        '''
        synchronous encoding
        '''
        self.encode(frames)
        self.finish()       # finish signal to end the dead loop
        self.run()

    def run(self):
        while not self.finish_sign or not self.frames.empty():
            if not self.frames.empty():
                t = self.frames.get()  # pop first
                if self.encode_frame(t):
                    if not self.save_frame():
                        raise RuntimeError('failed to save frame')
            else:
                time.sleep(T_SLEEP)
        self.flush()
        self.finished = True
        self.release()

    def encode_frame(self, frame: Frame) -> bool:
        '''
        encode a plane rgb frame
        the encoded frame should be assigned to self.enc_frame
        return true if the frame is ready
        '''
        raise NotImplementedError

    def save_frame(self) -> bool:
        '''
        convert the encoded frame into a byte stream
        save the encoded frame to the disk
        '''
        raise NotImplementedError

    def flush(self):
        '''
        [optional] encoder may be asynchronous
        '''
        pass

    def release(self):
        '''
        [optional] manually release resources
        '''
        pass

class BaseDecoder():
    def __init__(self, 
        file_name: str,
        id: Optional[Union[int, Tuple, str]]=None, # specify a id for decoder
        queue: Optional[Queue]=None     # share a same queue
    ):
        assert os.path.exists(file_name), f'{file_name} not exists'
        self.file_name = file_name
        self.id = id
        self.width = None
        self.height = None
        self.fps = None # frame per second
        self.nf = None # number of frames (may not work)
        self.dec_frame = None
        self.start_pos = 0
        self.idx = 0    # frame number in a gop
        self.queue = queue

    def get_frame(self) -> Union[Frame, torch.Tensor]:
        assert self.queue is None, 'use decode_all and get frames from the queue'
        if not self.decode_frame():
            return None
        self.idx += 1
        return self.dec_frame
    
    def curr_idx(self):
        return self.idx

    def offset(self):
        return self.idx - self.start_pos

    def set_start_pos(self, start_pos: int):
        '''
        set start position for decoding
        '''
        raise NotImplementedError

    def get_metadata(self):
        '''
        init width, height, fps and nf attributes
        '''
        raise NotImplementedError

    def decode_frame(self) -> bool:
        '''
        decode a frame to rgb plane format or pytorch tensor
        the decoded frame should be assigned to self.dec_frame
        return true if not empty, otherwise the decoding is finish
        '''
        raise NotImplementedError

    def skip_frame(self) -> bool:
        '''
        skip a frame to avoid decoding
        return true if not empty, otherwise the skiping is finish
        '''
        raise NotImplementedError

    def decode_all(self, limit: Optional[int]=None, skip_list: List[int]=[]):
        '''
        decode all frames to rgb plane format or pytorch tensor
        the decoded frame should be added into the queue
        it will finish after decoding `limit` frames
        '''
        assert self.id is not None, 'id is None'
        assert self.queue is not None, 'queue is None'
        cnt = 0
        skip_list = sorted(skip_list)
        skip_cnt = 0
        while limit is None or cnt < limit:
            while skip_cnt < len(skip_list) and self.curr_idx() > skip_list[skip_cnt]:
                skip_cnt += 1
            if skip_cnt < len(skip_list) and self.curr_idx() == skip_list[skip_cnt]:
                skip_cnt += 1
                if not self.skip_frame():
                    break
            else:
                if not self.decode_frame():
                    break
                self.queue.put((self.id, self.curr_idx(), self.dec_frame))
            self.idx += 1
            cnt += 1
        self.release()

    def empty_frame(self):
        raise NotImplementedError

    def release(self):
        '''
        [optional] manually release resources
        '''
        pass