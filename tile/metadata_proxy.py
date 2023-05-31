import json
import os, sys
sys.path.append('.')
from tile.utils import str2tuple, tuple2str
from tile.rect import str2rect, Rect
from typing import List, Tuple
Layout = List[List[Tuple]]
FrameRange = Tuple[int, int]

FILE_SYSTEM = 0
DATABASE = 1

class MetadataProxy:
    def __init__(self, engine: int=FILE_SYSTEM) -> None:
        self.engine = engine
        assert self.engine in [FILE_SYSTEM, DATABASE], \
            f'engine only support FILE_SYSTEM (0) or DATABASE (1)'
        self.metadata = {'layouts': {}, 'video': {}}
        self.frame_cnt = 0
        self.tile_cnt = 0
        self.tile_roi_cnt = 0

    def record_layout(self, enc_range: FrameRange, layout: Layout):
        s, e = enc_range
        duration = e - s
        rows, cols = len(layout), len(layout[0])
        metadata = {
            'duration': duration,
            'rows': rows, 'cols': cols,
            'roi': {}
        }
        self.frame_cnt += duration
        self.tile_cnt += duration * rows * cols

        N, M = len(layout), len(layout[0])
        for i in range(N):
            for j in range(M):
                rect, has_roi = layout[i][j][:2]
                if has_roi:
                    metadata['roi'][tuple2str((i, j))] = str(rect)
                    self.tile_roi_cnt += duration
                    if len(layout[i][j]) == 3: # additional info
                        if 'info' not in metadata:
                            metadata['info'] = {}
                        metadata['info'][tuple2str((i, j))] = layout[i][j][2]

        self.metadata['layouts'][tuple2str(enc_range)] = metadata

    def record_summarize(self, summarize: dict):
        for k, v in summarize.items():
            self.metadata['video'][k] = v

    def write(self, **param):
        self.record_summarize({
            'total_frames': self.frame_cnt, 
            'total_tiles': self.tile_cnt,
            'tiles_with_roi': self.tile_roi_cnt,
        })
        if self.engine == FILE_SYSTEM:
            self.write2fs(**param)
        elif self.engine == DATABASE:
            self.write2db(**param)
        else:
            raise TypeError(f'engine only support FILE_SYSTEM (0) or DATABASE (1)')

    def write2fs(self, root: str):
        '''
        This method is only a DEMO
        metadata should be record in a database
        '''
        metadata_path = os.path.join(root, 'metadata.json')
        assert not os.path.exists(metadata_path), f'{metadata_path} already exists'
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=4)

    def write2db(self):
        # TODO
        pass

    def read(self, path: str):
        if self.engine == FILE_SYSTEM:
            if os.path.isdir(path):
                metadata_path = os.path.join(path, 'metadata.json')
            else:
                metadata_path = path
            assert os.path.exists(metadata_path), f'{metadata_path} not exists'
            self.metadata = json.load(open(metadata_path, 'r'))
            self.layout_metadata = self.metadata['layouts']
            self.video_metadata = self.metadata['video']
        else:
            raise NotImplementedError
    
    # video info
    def count_frames(self) -> int:
        return self.video_metadata['total_frames']
    
    def count_tiles(self) -> int:
        return self.video_metadata['total_tiles']
    
    def count_roi_tiles(self) -> int:
        return self.video_metadata['tiles_with_roi']
    
    def get_resolution(self):
        h = self.video_metadata['height']
        w = self.video_metadata['width']
        return (h, w)
    
    def get_gop(self) -> int:
        return self.video_metadata['gop']

    # layout info
    def get_segments(self):
        segments = [str2tuple(x) for x in self.layout_metadata.keys()]
        segments = sorted(segments, key=lambda x:x[0])
        return segments
    
    def get_duration(self, enc_range: tuple):
        enc_range = tuple2str(enc_range)
        return self.layout_metadata[enc_range]['duration']
    
    def get_partition(self, enc_range: tuple):
        enc_range = tuple2str(enc_range)
        rows = self.layout_metadata[enc_range]['rows']
        cols = self.layout_metadata[enc_range]['cols']
        return (rows, cols)
    
    def get_tile_pos(self, enc_range: tuple):
        enc_range = tuple2str(enc_range)
        all_pos = list(self.layout_metadata[enc_range]['roi'].keys())
        return [str2tuple(x) for x in all_pos]
    
    def get_tile_rect(self, enc_range: tuple, tile_pos: tuple):
        enc_range = tuple2str(enc_range)
        tile_pos = tuple2str(tile_pos)
        return str2rect(self.layout_metadata[enc_range]['roi'][tile_pos])
    
    def get_tile_info(self, enc_range: tuple, tile_pos: tuple):
        enc_range = tuple2str(enc_range)
        tile_pos = tuple2str(tile_pos)
        return self.layout_metadata[enc_range]['info'][tile_pos]
    
    def get_sub_layout(self, enc_range: tuple, tile_pos: tuple):
        try:
            info = self.get_tile_info(enc_range, tile_pos)
            rect = self.get_tile_rect(enc_range, tile_pos)
            N, M = info['rows'], info['cols']
            xs, ys = {rect.x1, rect.x2}, {rect.y1, rect.y2}
            layout = [[None for _ in range(M)] for _ in range(N)]
            # roi-contained tiles
            for i in range(N):
                for j in range(M):
                    sub_tile_pos = tuple2str((i, j))
                    if sub_tile_pos in info['roi']:
                        tile = str2rect(info['roi'][sub_tile_pos])
                        layout[i][j] = (tile, True)
                        xs |= {tile.x1, tile.x2}
                        ys |= {tile.y1, tile.y2}
            xs, ys = sorted(list(xs)), sorted(list(ys))
            for i in range(N):
                for j in range(M):
                    if not layout[i][j]:
                        tile = Rect(xs[j], ys[i], xs[j+1], ys[i+1])
                        layout[i][j] = (tile, False)
            return layout
        except:
            return None

if __name__ == '__main__':
    proxy = MetadataProxy()
    proxy.read('out/hier.json')
    print(proxy.get_sub_layout((8280, 8310), (1, 5)))