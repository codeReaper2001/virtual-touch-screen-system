from typing import List
from . import hand_tracking


def flatten_data(lm_list: List[hand_tracking.LmData]):
    res: List[float] = []
    for lm in lm_list:
        res.extend([lm.x, lm.y, lm.z])
    return res

def fingerlist_to_finger_bitmap(fingers: List[bool]):
    res = 0
    for id, finger in enumerate(fingers):
        if finger:
            res += 1 << (4 - id)
    return res
