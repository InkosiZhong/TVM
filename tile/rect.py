from typing import Tuple, List
Point = Tuple[float, float]

def float2int(pt: Point) -> Point:
    x, y = pt
    return (int(x), int(y))

def distance(pt1: Point, pt2: Point) -> float:
    x1, y1 = pt1
    x2, y2 = pt2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

class Rect:
    '''
    This is a pure data class, so the value of it cannot be modified after initialization
    If a modify happened, you should construct a new object
    '''
    def __init__(self, x1:float, y1:float, x2:float=None, y2:float=None, w:float=None, h:float=None, scale_factor=1) -> None:
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2 if x2 != None else x1 + w
        self.y2 = y2 if y2 != None else y1 + h
        if self.x1 > self.x2:
            self.x1, self.x2 = self.x2, self.x1
        if self.y1 > self.y2:
            self.y1, self.y2 = self.y2, self.y1
        self.x1 *= scale_factor
        self.y1 *= scale_factor
        self.x2 *= scale_factor
        self.y2 *= scale_factor
        self.w = self.x2 - self.x1
        self.h = self.y2 - self.y1

    def xyxy(self, dtype=float) -> tuple:
        return (dtype(self.x1), dtype(self.y1), dtype(self.x2), dtype(self.y2))

    def xywh(self, dtype=float) -> tuple:
        return (dtype(self.x1), dtype(self.y1), dtype(self.w), dtype(self.h))
    
    def lt(self, dtype=float) -> Point:
        '''
        Left-Top cornor point
        '''
        return (dtype(self.x1), dtype(self.y1))

    def rt(self, dtype=float) -> Point:
        '''
        Right-Top cornor point
        '''
        return (dtype(self.x2), dtype(self.y1))

    def lb(self, dtype=float) -> Point:
        '''
        Left-Bottom cornor point
        '''
        return (dtype(self.x1), dtype(self.y2))

    def rb(self, dtype=float) -> Point:
        '''
        Right-Bottom cornor point
        '''
        return (dtype(self.x2), dtype(self.y2))

    def ctr(self) -> Point:
        '''
        Center point
        '''
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def diagonal(self) -> float:
        '''
        length of Diagonal 
        '''
        return distance(self.lt(), self.rb())

    def area(self) -> float:
        return self.w * self.h

    def contain(self, pt: Point) -> bool:
        x, y = pt
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def include(self, rect) -> bool:
        return self.contain(rect.lt()) and self.contain(rect.rb())
        
    def __repr__(self) -> str:
        return f'({self.x1}, {self.y1}, {self.w}, {self.h})'

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, Rect):
            return self.x1 == __value.x1 and \
                self.x2 == __value.x2 and \
                self.y1 == __value.y1 and \
                self.y2 == __value.y2
        return False
    
    def __deepcopy__(self, memo):
        new_rect = Rect(self.x1, self.y1, self.x2, self.y2)
        return new_rect

def str2rect(s: str) -> Rect:
    tuple = s.strip('(').strip(')').split(', ')
    x1, y1, w, h = tuple
    return Rect(float(x1), float(y1), w=float(w), h=float(h))

def intersect(rect1: Rect, rect2: Rect) -> bool:
    rect = merge_rect(rect1, rect2)
    return rect.w <= rect1.w + rect2.w and rect.h <= rect1.h + rect2.h
    '''return rect1.contain(rect2.lt()) or rect1.contain(rect2.rt()) or \
        rect1.contain(rect2.lb()) or rect1.contain(rect2.rb()) or \
        rect2.contain(rect1.lt()) or rect2.contain(rect1.rt()) or \
        rect2.contain(rect1.lb()) or rect2.contain(rect1.rb())'''

def intersect_rect(rect1: Rect, rect2: Rect) -> Rect:
    if not intersect(rect1, rect2):
        return None
    x1 = max(rect1.x1, rect2.x1)
    x2 = min(rect1.x2, rect2.x2)
    y1 = max(rect1.y1, rect2.y1)
    y2 = min(rect1.y2, rect2.y2)
    if x1 >= x2 or y1 >= y2:
        return None
    return Rect(x1, y1, x2, y2)

def IoU(rect1: Rect, rect2: Rect) -> float:
    inter = intersect_rect(rect1, rect2)
    i = inter.area() if inter is not None else 0
    return i / (rect1.area() + rect2.area() - i)

def ctr_distance(rect1: Rect, rect2: Rect) -> float:
    '''
    distance between center point
    '''
    return distance(rect1.ctr(), rect2.ctr())

VERTICAL = 1
HORIZONTAL = 2
def edge_distance(rect1: Rect, rect2: Rect) -> Tuple[float, int]:
    '''
    min edge distance between rects
    '''
    raise RuntimeError('this function has bug need to be fixed')
    d_vertical = min(
        abs(rect1.y1 - rect2.y1),
        abs(rect1.y1 - rect2.y2),
        abs(rect1.y2 - rect2.y1),
        abs(rect1.y2 - rect2.y2)
    )
    d_horizontal = min(
        abs(rect1.x1 - rect2.x1), 
        abs(rect1.x1 - rect2.x2),
        abs(rect1.x2 - rect2.x1),
        abs(rect1.x2 - rect2.x2)
    )
    if d_vertical < d_horizontal:
        direct = VERTICAL
        d = d_vertical
    elif d_horizontal < d_vertical:
        direct = HORIZONTAL
        d = d_horizontal
    else:
        direct = VERTICAL | HORIZONTAL
        d = d_vertical
    return (d, direct)

def merge_rect(rect1: Rect, rect2: Rect) -> Rect:
    return Rect(min(rect1.x1, rect2.x1),
                min(rect1.y1, rect2.y1),
                max(rect1.x2, rect2.x2),
                max(rect1.y2, rect2.y2))

def merge_multi_rect(rects: List[Rect]) -> Rect:
    return Rect(min([rect.x1 for rect in rects]),
                min([rect.y1 for rect in rects]),
                max([rect.x2 for rect in rects]),
                max([rect.y2 for rect in rects]))

def exclude(src: Rect, rect: Rect) -> List[Rect]:
    assert src.include(rect), f'exclude overflow: {src} -> {rect}'
    rects = [
        Rect(*src.lt(), *rect.lt()), Rect(*rect.lt(), rect.x2, src.y1), Rect(*rect.rt(), *src.rt()),
        Rect(*rect.lt(), src.x1, rect.y2), Rect(*rect.rt(), src.x2, rect.y2),
        Rect(*src.lb(), *rect.lb()), Rect(*rect.lb(), rect.x2, src.y2), Rect(*rect.rb(), *src.rb()),
    ]
    return [x for x in rects if x.area() > 0]

if __name__ == '__main__':
    print(intersect_rect(Rect(0, 0, 2, 2), Rect(0, 0, 1, 1)))
    print(intersect_rect(Rect(0, 0, 1, 1), Rect(1, 1, 2, 2)))
    print(intersect_rect(Rect(0, 0, 2, 2), Rect(1, 1, 3, 3)))
    print(intersect_rect(Rect(0, 1, 2, 3), Rect(1, 0, 3, 2)))