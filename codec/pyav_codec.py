try:
    from base_codec import BaseEncoder, Frame
    from config import PyAVEncoderConfig
except:
    from .base_codec import BaseEncoder, Frame
    from .config import PyAVEncoderConfig
import av
import time

class PyAVEncoder(BaseEncoder):
    def __init__(self, file_name: str, cfg: PyAVEncoderConfig):
        super().__init__(file_name, cfg)
        self.container = av.open(file_name, mode="w")
        self.stream = self.container.add_stream(cfg.codec, rate=cfg.fps, 
            options=cfg.to_map())
        #self.stream.thread_type = 'AUTO'
        #self.stream.thread_count = 1
        self.stream.pix_fmt = cfg.pix_fmt

    def dynamic_init(self, frame: Frame):
        height, width = frame.shape[:2]
        self.stream.width = width
        self.stream.height = height
        super().dynamic_init(frame)

    def encode_frame(self, frame: Frame) -> bool:
        frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        self.enc_frame = self.stream.encode(frame)
        return len(self.enc_frame) > 0

    def save_frame(self) -> bool:
        for packet in self.enc_frame:
            self.container.mux(packet)
        return True

    def flush(self):
        self.enc_frame = self.stream.encode()
        self.save_frame()

    def release(self):
        self.container.close()

if __name__ == '__main__':
    # sync style
    from cv2_codec import CV2Decoder
    from queue import Queue
    import os
    if os.path.exists('out.mp4'):
        os.remove('out.mp4')
    q = Queue()
    enc = PyAVEncoder('out.mp4', PyAVEncoderConfig())
    dec = CV2Decoder('data/270p.mp4', 0, q)
    dec.set_start_pos(15)
    dec.decode_all(30)
    cnt = 0
    frames = []
    while not q.empty():
        cnt += 1
        id, idx, frame = q.get()
        print(id, idx)
        frames.append(frame)
    enc.sync_encode(frames)
    print(cnt)