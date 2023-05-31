from typing import Callable
try:
    from rect import *
except:
    from .rect import *
from functools import partial

ConditionFunc = Callable[[Rect, Rect], bool] # return True if satisfy the condition

def template_ctr_abs_close_condition(rect1: Rect, rect2: Rect, threashold: float) -> bool:
    '''
    Judge if the absolute distance between the Rects' center point are close
    '''
    d = ctr_distance(rect1, rect2)
    return d <= threashold

def create_ctr_abs_close_condition(threashold: float) -> ConditionFunc:
    return partial(template_ctr_abs_close_condition, threashold=threashold)

def template_ctr_rel_close_condition(rect1: Rect, rect2: Rect, threashold: float) -> bool:
    '''
    Judge if the relative distance between the Rects' center point are close
    '''
    d = ctr_distance(rect1, rect2)
    return d <= threashold * min(rect1.diagonal(), rect2.diagonal())

def create_ctr_rel_close_condition(threashold: float) -> ConditionFunc:
    return partial(template_ctr_rel_close_condition, threashold=threashold)

def template_edge_abs_close_condition(rect1: Rect, rect2: Rect, threashold: float) -> bool:
    '''
    Judge if the absolute distance between the Rects' edge are close
    '''
    d, _ = edge_distance(rect1, rect2)
    return d < threashold

def create_edge_abs_close_condition(threashold: float) -> ConditionFunc:
    return partial(template_edge_abs_close_condition, threashold=threashold)

def template_edge_rel_close_condition(rect1: Rect, rect2: Rect, threashold: float) -> bool:
    '''
    Judge if the relative distance between the Rects' edge are close
    '''
    d, direct = edge_distance(rect1, rect2)
    d_base = 0
    if direct & VERTICAL:
        d_base = max(rect1.h, rect2.h)
    if direct & HORIZONTAL:
        d_base = max(d_base, rect1.w, rect2.w)
    return d < threashold * d_base

def create_edge_rel_close_condition(threashold: float) -> ConditionFunc:
    return partial(template_edge_rel_close_condition, threashold=threashold)