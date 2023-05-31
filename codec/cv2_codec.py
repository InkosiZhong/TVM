try:
    from base_codec import BaseDecoder
except:
    from .base_codec import BaseDecoder
import cv2
import numpy as np
import time
from queue import Queue
from typing import Optional, Union, Tuple

class CV2Decoder(BaseDecoder):
    def __init__(self, 
        file_name: str, 
        id: Optional[Union[int, Tuple, str]]=None, 
        queue: Queue=None
    ) -> None:
        super().__init__(file_name, id, queue)
        self.cap = cv2.VideoCapture(file_name)
        timeout = 1 # 1s
        while not self.cap.isOpened():
            time.sleep(1e-3)
            timeout -= 1e-3
            if timeout <= 0:
                raise TimeoutError(f'CV2Decoder cannot load video {file_name}')
        self.get_metadata()

    def get_metadata(self):
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.nf = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

    def set_start_pos(self, start_pos: int) -> bool:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_pos)
        if self.cap.get(cv2.CAP_PROP_POS_FRAMES) == start_pos:
            self.start_pos = start_pos
            self.idx = start_pos
            return True
        return False

    def decode_frame(self) -> bool:
        valid, raw_frame = self.cap.read()
        if valid:
            self.dec_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
        return valid

    def skip_frame(self) -> bool:
        if self.cap.grab():
            self.idx += 1
            return True
        return False
    
    def empty_frame(self):
        return np.zeros(self.height, self.width, 3)

    def release(self):
        self.cap.release()

if __name__ == '__main__':
    from pyav_codec import PyAVEncoder
    from config import PyAVEncoderConfig
    import os, time
    dec = CV2Decoder('data/270p.mp4')
    print(dec.width, dec.height, dec.fps, dec.nf)
    if os.path.exists('out.mp4'):
        os.remove('out.mp4')
    enc = PyAVEncoder('out.mp4', PyAVEncoderConfig())
    enc.start()
    cnt = 0
    dec_cnt = 0
    while True:
        if cnt % 10 == 0:
            if not dec.skip_frame():
                break
        else:
            frame = dec.get_frame()
            if frame is None:
                break
            enc.encode([frame])
            dec_cnt += 1
        cnt += 1
    enc.finish()
    enc.join() # wait for done
    print(f'transcode {dec_cnt} frames')