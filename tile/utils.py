from typing import Any, List, Tuple, Callable
import os
import time
from threading import Lock

def tuple2str(t: tuple) -> str:
    return '_'.join([str(x) for x in t])

def str2tuple(s: str, val_type: type=int) -> tuple:
    return tuple([val_type(x) for x in s.split('_')])

br_map = {
    'k': 1e3, 'm': 1e6, 'g': 1e9
}
def br_str2int(br: str) -> int:
    num, base = int(br[:-1]), br[-1]
    return num * br_map[base]

def br_float2str(br: int) -> str:
    for k, v in br_map.items():
        if br > v:
            br = round(br / v)
            return f'{br}{k}'

def binary_search(arr: list, x: Any, hit_func: Callable[[Any, Any], int]=lambda x, y: x-y) -> int:
    def _search(l: int, r: int):
        if r <= l:
            return None
        mid = (l + r) // 2
        hit = hit_func(x, arr[mid])
        if hit == 0:
            return mid
        elif hit > 0:
            return _search(mid + 1, r)
        else:
            return _search(l, mid)
    return _search(0, len(arr))

def hit_in(x: Any, y: Tuple[Any, Any]) -> int:
    s, e = y
    if x < s:
        return -1
    elif x >= e:
        return 1
    else:
        return 0

def get_tile_path(root, dec_range, tile_pos):
    return os.path.join(root, tuple2str(dec_range), tuple2str(tile_pos) + '.mp4')

def union_find(x, parent):
    while x != parent[x]:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union_join(x, y, parent):
    p1 = union_find(x, parent)
    p2 = union_find(y, parent)
    if p1 != p2:
        parent[p1] = p2

def union(items: list, condition: Callable):
    parent = [i for i in range(len(items))]
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if condition(items[i], items[j]):
                union_join(i, j, parent)
    return parent

class BaseBuffer:
    def __init__(self, max_size: int=1000, mutex: bool=False) -> None:
        self.max_size = max_size
        self.buf = {}
        self.available = {}
        self.mutex = mutex
        self.lock = Lock()
    
    def cache(self, key: Any, value: Any):
        if self.max_size == 0:
            return
        num = self.update(key)
        if num > self.max_size:
            dis_key = self.discard()
            self.buf.pop(dis_key)
            if self.mutex:
                self.available.pop(dis_key)
        self.buf[key] = value # replace
        if self.mutex:
            self.available[key] = True

    def manual_discard(self, key: Any):
        if key in self.buf:
            self.discard(key)
            self.buf.pop(key)
            if self.mutex:
                self.available[key] = False

    def __setitem__(self, key: Any, value: Any):
        if value is None:
            self.manual_discard(key)
        else:
            self.cache(key, value)

    def get_cache(self, key: Any) -> Any:
        if key in self.buf:
            if self.mutex:
                if self.available[key]:
                    self.available[key] = False
                else:
                    return None
            return self.buf[key]
        return None
    
    def __getitem__(self, key: Any) -> Any:
        if self.mutex:
            with self.lock:
                return self.get_cache(key)
        else:
            return self.get_cache(key)
        
    def __contains__(self, key: Any) -> bool:
        return key in self.buf

    def size(self) -> int:
        return len(self.buf)

    def __len__(self) -> int:
        return self.size()

    def discard(self, key: Any=None) -> Any:
        raise NotImplementedError

    def update(self, key: Any) -> int:
        raise NotImplementedError

class LRUBuffer(BaseBuffer):
    def __init__(self, max_size: int=1000, mutex: bool=False) -> None:
        super().__init__(max_size, mutex)
        '''
        WARNING: ONLY SUPPORT FOR Python >= 3.7
        Since python do not support pointer, this is a temporary solution.
        '''
        self.stack = {} # 0 as bottom, -1 as top
    
    def btm_key(self) -> Any:
        return next(iter(self.stack.keys()))
    
    def discard(self, key: Any=None) -> Any:
        if key is None:
            return self.stack.pop(self.btm_key())
        else:
            return self.stack.pop(key)

    def update(self, key: Any) -> int:
        if key in self.stack:
            self.stack.pop(key)
        self.stack[key] = key
        return len(self.stack)
    
from collections import deque
class DequeLRUBuffer(BaseBuffer):
    def __init__(self, max_size: int=1000, mutex: bool=False) -> None:
        super().__init__(max_size, mutex)
        self.stack = deque() # 0 as bottom, -1 as top
    
    def discard(self, key: Any=None) -> Any:
        if key is None:
            return self.stack.popleft()
        else:
            self.stack.remove(key)

    def update(self, key: Any) -> int:
        if key in self.stack:
            self.stack.remove(key)
        self.stack.append(key) # append at top
        return len(self.stack)

if __name__ == '__main__':
    buf = LRUBuffer(5)
    input_list = [1, 2, 3, 4, 5, 5, 1, 6, 7, 4, 9, 3, 4, 7]
    for i in input_list:
        print(buf[i])
        buf[i] = i