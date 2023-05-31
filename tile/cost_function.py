from typing import Callable
from functools import partial

CostFunc = Callable[[float], float] # area -> cost

def template_ra_cost(num_pix: int, codec: dict, seg: int, gop: int) -> float:
    ''' Random access for video decoder
    If we split a video into segments with length S,
    seeking will only happens when the target frame is not in the first GOP.
    So, we use p = 1 - L / S to model the possibility
    The eq.1 should be modified as:
        t1 = t_init + t_seek * p + t_skip * (L / 2 - 1) + t_dec + t_rel
    '''
    n_skip = (gop / 2) - 1
    p_seek = 1 - gop / seg
    return codec['k_init'] * num_pix + codec['b_init'] + \
        (codec['k_seek'] * num_pix + codec['b_seek']) * p_seek + \
        (codec['k_skip'] * num_pix + codec['b_skip']) * n_skip + \
        (codec['k_dec'] * num_pix + codec['b_dec']) + \
        codec['k_rel'] * num_pix + codec['b_rel']

def create_codec_cost_func(codec: dict, gop: int) -> CostFunc:
    return partial(template_ra_cost, codec=codec, seg=gop, gop=gop)

def template_dnn_cost(num_pix: int, dnn: dict, inflection: float) -> float:
    if num_pix > inflection:
        return dnn['k'] * num_pix + dnn['b']
    else:
        return dnn['a']

def create_dnn_cost_func(dnn: dict) -> CostFunc:
    inflection = (dnn['a'] - dnn['b']) / dnn['k']
    print(f'inflection: {inflection}')
    return partial(template_dnn_cost, dnn=dnn, inflection=inflection)