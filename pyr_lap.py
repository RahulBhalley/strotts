import torch
import torch.nn.functional as F

def dec_lap_pyr(X, levs):
    pyr = []
    cur = X
    for _ in range(levs):
        cur_x, cur_y = cur.size(2), cur.size(3)
        x_small = F.interpolate(cur, scale_factor=0.5, mode='bilinear', align_corners=False)
        x_back = F.interpolate(x_small, size=(cur_x, cur_y), mode='bilinear', align_corners=False)
        lap = cur - x_back
        pyr.append(lap)
        cur = x_small
    pyr.append(cur)
    return pyr

def syn_lap_pyr(pyr):
    cur = pyr[-1]
    for lap in reversed(pyr[:-1]):
        cur = lap + F.interpolate(cur, size=(lap.size(2), lap.size(3)), mode='bilinear', align_corners=False)
    return cur