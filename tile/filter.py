from typing import Callable, List, Tuple
from codec import Size
try:
    from rect import *
except:
    from .rect import *
from functools import partial

FilterFunc = Callable[[Rect], bool]     # return True if should be prune

def template_area_filter(rect: Rect, skip_area: List[Rect], threashold: float) -> bool:
    for area in skip_area:
        inter = intersect_rect(area, rect)
        if inter is not None and inter.area() > threashold * rect.area():
            return True
    return False

def create_area_filter(skip_area: List[Rect], threashold: float=0.3) -> FilterFunc:
    return partial(template_area_filter, skip_area=skip_area, threashold=threashold)

def template_size_filter(rect: Rect, min_size: Size, max_size: Size):
    if min_size != None:
        min_h, min_w = min_size
        if rect.w < min_w or rect.h < min_h:
            return True
    if max_size != None:
        max_h, max_w = max_size
        if rect.w > max_w or rect.h > max_h:
            return True
    return False

def create_size_filter(min_size: Size, max_size: Size=None):
    return partial(template_size_filter, min_size=min_size, max_size=max_size)
    
if __name__ == '__main__':
    skip_area = [Rect(50, 50, w=50, h=50)]
    roi1 = Rect(50, 50, 10, 10)
    roi2 = Rect(0, 0, 10, 10)

    print(template_area_filter(roi1, skip_area))
    print(template_area_filter(roi2, skip_area))

    area_filter = create_area_filter(skip_area)
    print(area_filter(roi1))
    print(area_filter(roi2))