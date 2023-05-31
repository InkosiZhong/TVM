import sys
import os
from queue import Queue
from typing import Optional, Union, Tuple

if os.name == 'nt':
    # Add CUDA_PATH env variable
    cuda_path = os.environ["CUDA_PATH"]
    if cuda_path:
        os.add_dll_directory(cuda_path)
    else:
        print("CUDA_PATH environment variable is not set.", file=sys.stderr)
        print("Can't set CUDA DLLs search path.", file=sys.stderr)
        exit(1)

    # Add PATH as well for minor CUDA releases
    sys_path = os.environ["PATH"]
    if sys_path:
        paths = sys_path.split(';')
        for path in paths:
            if os.path.isdir(path):
                os.add_dll_directory(path)
    else:
        print("PATH environment variable is not set.", file=sys.stderr)
        exit(1)

import pycuda.driver as cuda
cuda.init()

import numpy as np
import torch, torchvision
import PyNvCodec as nvc
import PytorchNvCodec as pnvc
import av   # PyAV is only used for muxing
try:
    from base_codec import BaseEncoder, BaseDecoder, Frame, Size
    from config import VPFEncoderConfig
    from color_converter import ColorConverter
except:
    from .base_codec import BaseEncoder, BaseDecoder, Frame, Size
    from .config import VPFEncoderConfig
    from .color_converter import ColorConverter

from .config import PyAVEncoderConfig
FFMPEG_TBN = 15360
class VPFEncoder(BaseEncoder):
    def __init__(self, file_name: str, cfg: VPFEncoderConfig, gpu_id:int=0):
        super().__init__(file_name, cfg)
        # retain primary CUDA device context and create separate stream per thread.
        self.gpu_id = gpu_id
        self.ctx = cuda.Device(self.gpu_id).retain_primary_context()
        self.ctx.push()
        self.str = cuda.Stream()
        self.ctx.pop()
        self.enc_frame = np.ndarray(shape=(0), dtype=np.uint8)
        self.container = av.open(file_name, 'w')
        self.stream = self.container.add_stream(cfg.codec, rate=cfg.fps, options=PyAVEncoderConfig().to_map())
        self.cnt = 0
        self.step = FFMPEG_TBN / cfg.fps

    def dynamic_init(self, frame: Frame):
        height, width = frame.shape[:2]
        self.nv_uploader = nvc.PyFrameUploader(width, height, nvc.PixelFormat.RGB,
                                         self.ctx.handle, self.str.handle)

        self.to_nv12 = ColorConverter(width, height, self.gpu_id)
        self.to_nv12.add(nvc.PixelFormat.RGB, nvc.PixelFormat.YUV420)
        self.to_nv12.add(nvc.PixelFormat.YUV420, nvc.PixelFormat.NV12)

        cfg_map = self.cfg.to_map()
        cfg_map['s'] = str(width) + 'x' + str(height)
        self.nv_enc = nvc.PyNvEncoder(cfg_map, self.ctx.handle, self.str.handle)

        self.stream.width = width
        self.stream.height = height

        super().dynamic_init(frame)

    def encode_frame(self, frame: Frame) -> bool:
        raw_surface = self.nv_uploader.UploadSingleFrame(frame)
        if raw_surface.Empty():
            raise RuntimeError('Failed to upload video frame to GPU.')
        cvt_surface = self.to_nv12.run(raw_surface)
        if cvt_surface.Empty():
            raise RuntimeError('Failed to do color conversion.')
        return self.nv_enc.EncodeSingleSurface(cvt_surface, self.enc_frame)

    def save_frame(self) -> bool:
        enc_array = bytearray(self.enc_frame)
        packet = av.packet.Packet(enc_array)
        packet.stream = self.stream
        packet.dts = self.cnt
        packet.pts = self.cnt
        self.cnt += self.step
        self.container.mux_one(packet)
        '''bin_array = bytearray(self.enc_frame)
        with open(self.file_name, 'ab') as f:
            f.write(bin_array)'''
        return True

    def flush(self):
        #if self.nv_enc.Flush(self.enc_frame):
        while self.nv_enc.FlushSinglePacket(self.enc_frame):
            if not self.save_frame():
                raise RuntimeError('Failed to save flushed frame')

    def release(self):
        self.container.close()

class SurfaceConverter:
    def __init__(self,
        raw_size: Size,
        target_size: Size=None, # (height, width)
        frame_type: type=Frame,
        gpu_id: int=0,
        # only need when frame_type is Frame, set None will re-create
        cuda_ctx=None,
        cuda_str=None
    ):
        raw_height, raw_width = raw_size
        self.target_size = target_size
        self.frame_type = frame_type
        self.ctx = cuda_ctx
        self.str = cuda_str
        if self.target_size is not None:
            h, w = self.target_size
            assert 0 < h and 0 < w, 'invalid target_size'
        else:
            h, w = raw_height, raw_width
        self.shape = (h, w, 3)
        self.to_yuv = ColorConverter(raw_width, raw_height, gpu_id)
        self.to_yuv.add(nvc.PixelFormat.NV12, nvc.PixelFormat.YUV420)
        if self.target_size is not None:
            self.to_dim = nvc.PySurfaceResizer(w, h, nvc.PixelFormat.YUV420, gpu_id)
        self.to_rgb = ColorConverter(w, h, gpu_id)
        self.to_rgb.add(nvc.PixelFormat.YUV420, nvc.PixelFormat.RGB)
        if self.frame_type is Frame: # cpu
            if self.ctx is None or self.str is None:
                self.ctx = cuda.Device(gpu_id).retain_primary_context()
                self.ctx.push()
                self.str = cuda.Stream()
                self.ctx.pop()
            self.nv_downloader = nvc.PySurfaceDownloader(w, h, self.to_rgb.dst_fmt, self.ctx.handle, self.str.handle)
        elif self.frame_type is torch.Tensor: # gpu
            self.to_rgb.add(nvc.PixelFormat.RGB, nvc.PixelFormat.RGB_PLANAR)
        else:
            raise TypeError(f'{self.frame_type} is not supported, using Frame or torch.Tensor')
            
    def __call__(self, raw_surface):
        assert not raw_surface.Empty(), 'input an empty surface'
        cvt_surface = self.to_yuv.run(raw_surface)
        if self.target_size is not None:
            cvt_surface = self.to_dim.Execute(cvt_surface)
        cvt_surface = self.to_rgb.run(cvt_surface)
        if self.frame_type is Frame:
            raw_frame = np.ndarray(shape=(cvt_surface.HostSize()), dtype=np.uint8)
            if not self.nv_downloader.DownloadSingleSurface(cvt_surface, raw_frame):
                raise RuntimeError('Failed to download surface')
            dec_frame = raw_frame.reshape(self.shape)
        else:
            surf_plane = cvt_surface.PlanePtr()
            img_tensor = pnvc.makefromDevicePtrUint8(surf_plane.GpuMem(),
                                                    surf_plane.Width(),
                                                    surf_plane.Height(),
                                                    surf_plane.Pitch(),
                                                    surf_plane.ElemSize())

            img_tensor.resize_(3, *self.shape[:2])
            img_tensor = img_tensor.type(dtype=torch.cuda.FloatTensor)
            img_tensor = torch.divide(img_tensor, 255.0)
            #dec_frame = self.transform(img_tensor)
            dec_frame = img_tensor
        return dec_frame

class VPFDecoder(BaseDecoder):
    def __init__(self, 
        file_name: str,
        id: Optional[Union[int, Tuple, str]]=None,
        queue: Optional[Queue]=None,
        target_size: Size=None, # (height, width)
        frame_type: type=Frame,
        gpu_id: int=0
    ) -> None:
        super().__init__(file_name, id, queue)
        self.target_size = target_size
        self.frame_type = frame_type
        self.gpu_id = gpu_id
        # retain primary CUDA device context and create separate stream per thread.
        self.ctx = cuda.Device(gpu_id).retain_primary_context()
        self.ctx.push()
        self.str = cuda.Stream()
        self.ctx.pop()
        # create Decoder with given CUDA context & stream.
        self.nv_dec = nvc.PyNvDecoder(file_name, self.ctx.handle, self.str.handle)
        self.get_metadata()
        self.surf_cvt = SurfaceConverter((self.height, self.width), target_size, frame_type, gpu_id, self.ctx, self.str)
        self.seek_ctx = None
        
    def get_metadata(self):
        self.width, self.height = self.nv_dec.Width(), self.nv_dec.Height()
        self.fps = self.nv_dec.Framerate()
        self.nf = self.nv_dec.Numframes()

    def decode_frame(self) -> bool:
        raw_surface = self._get_raw_surface()
        if raw_surface.Empty():
            return False
        self.dec_frame = self.surf_cvt(raw_surface)
        return True

    def skip_frame(self) -> bool:
        return self.get_raw_surface() is not None
    
    def empty_frame(self):
        if self.target_size is not None:
            h, w = self.target_size
        else:
            h, w = self.height, self.width
        if self.frame_type is Frame:
            return np.zeros(h, w, 3)
        else:
            return torch.zeros(3, h, w).cuda(self.gpu_id)

    def get_raw_surface(self):
        raw_surface = self._get_raw_surface()
        if not raw_surface.Empty():
            self.idx += 1
            return raw_surface
        return None

    def _get_raw_surface(self):
        if self.seek_ctx is not None:
            raw_surface = self.nv_dec.DecodeSingleSurface(self.seek_ctx)
            self.seek_ctx = None
        else:
            raw_surface = self.nv_dec.DecodeSingleSurface()
        return raw_surface

    def set_start_pos(self, start_pos: int) -> bool:
        # FIXME Seek isn't supported for this input.
        self.seek_ctx = nvc.SeekContext(
            seek_frame=start_pos, seek_criteria=nvc.SeekCriteria.BY_NUMBER)
        self.start_pos = start_pos
        self.idx = start_pos
        return True

if __name__ == '__main__':
    import os
    q = Queue()
    dec = VPFDecoder('./data/270p.mp4', 0, q)#, target_size=[244, 244], frame_type=torch.Tensor)
    dec.set_start_pos(15)
    print(dec.width, dec.height, dec.fps, dec.nf)
    dec.decode_all(skip_list=[0, 14, 15, 149])
    frames = []
    while not q.empty():
        id, idx, frame = q.get()
        frames.append(frame)
        print(idx)
    print(f'decoded {len(frames)} frames')
    print(frames[0].shape)
    '''if os.path.exists('out.mp4'):
        os.remove('out.mp4')
    cfg = VPFEncoderConfig()
    enc = VPFEncoder('./out.mp4', cfg)
    enc.sync_encode(frames)'''