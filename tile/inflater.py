from typing import Callable, List, Tuple
import sys
sys.path.append('.')
from codec import Size
try:
    from rect import *
except:
    from .rect import *
from functools import partial

InflateFunc = Callable[[Rect], Rect]     # a x->x mapping to inflate the rect

def template_linear_inflater(rect: Rect, k: float) -> Rect:
    w, h = rect.w * k, rect.h * k
    x, y = rect.ctr()
    return Rect(int(x-w/2), int(y-h/2), w=int(w), h=int(h))

def create_linear_inflater(k: float) -> InflateFunc:
    return partial(template_linear_inflater, k=k)

def template_exp_inflater(rect: Rect, base: float, k: float) -> Rect:
    '''
    x = (k * base^(-x) + 1) * x, where x can be width and height
    '''
    w, h = rect.w, rect.h
    w, h = (k * base ** (-w) + 1) * w, (k * base ** (-h) + 1) * h
    x, y = rect.ctr()
    return Rect(int(x-w/2), int(y-h/2), w=int(w), h=int(h))

def create_exp_inflater(base: float, k: float) -> InflateFunc:
    return partial(template_exp_inflater, base=base, k=k)

def template_naive_inflater(rect: Rect, min_size: int) -> Rect:
    w, h = rect.w, rect.h
    x, y = rect.ctr()
    if w < min_size:
        x1, w = int(x-min_size/2), min_size
    else:
        x1 = rect.x1
    if h < min_size:
        y1, h = int(y-min_size/2), min_size
    else:
        y1 = rect.y1
    return Rect(x1, y1, w=w, h=h)

def create_naive_inflater(min_size: int) -> InflateFunc:
    return partial(template_naive_inflater, min_size=min_size)

if __name__ == '__main__':
    rect = create_linear_inflater(0.2)(Rect(1, 1, 2, 2))
    print(rect)
    rect = create_exp_inflater(1.05, 2)(Rect(1, 1, 2, 2))
    print(rect)
    rect = create_naive_inflater(2)(Rect(1, 1, 2, 2))
    print(rect)