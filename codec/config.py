from typing import Dict

class BaseEncoderConfig:
    def __init__(self) -> None:
        pass

    def is_attr(self, name: str) -> bool:
        '''
        customize to filter the functions and helper attrbutes
        '''
        return name[:2] not in ['__', 'to', 'is']

    def is_valid(self, name: str) -> bool:
        '''
        customize to filter the undefined attributes
        '''
        return self.__dict__[name] != None

    def to_map(self) -> Dict[str, str]:
        attr_names = [
            x for x in self.__dir__() if self.is_attr(x)
        ]
        cfg_map = {
            k: str(self.__dict__[k]) for k in attr_names 
                if self.is_valid(k)
        }
        return cfg_map

class PyAVEncoderConfig(BaseEncoderConfig):
    def __init__(self) -> None:
        super().__init__
        self.codec = 'libx265'     # [libx265, hevc, h264, hevc_nvenc, h264_nvenc]
        # set preset to None to enable RA mode
        self.preset = 'fast'    # [ultrafast, superfast, veryfast, faster, fast, medium, slow, veryslow]
        self.bf = 0             # number of B-frames
        self.crf = None
        self.fps = 30
        self.g = 30
        self.b = None           # '2000k'
        self.pix_fmt = 'yuv420p'

    def is_valid(self, name: str) -> bool:
        return name in ['bf', 'preset', 'crf', 'g', 'b'] \
            and self.__dict__[name] != None

class VPFEncoderConfig(BaseEncoderConfig):
    def __init__(self) -> None:
        super().__init__()
        self.codec = 'hevc'     # [hevc, h264]
        self.preset = 'P3'      # [P1, ..., P7]
        self.tuning_info = None
        self.profile = None
        self.bf = None          # number of B-frames
        self.rc = None          # rate-control method
        self.bitrate = None     # bitrate for 1s (rc=cbr/vbr)
        self.constqp = None     # [1, ..., 51] (rc=constqp)
        self.fps = 30
        self.gop = 30
            
if __name__ == '__main__':
    cfg = PyAVEncoderConfig()
    print(cfg.to_map()) 
    cfg = VPFEncoderConfig()
    print(cfg.to_map()) 