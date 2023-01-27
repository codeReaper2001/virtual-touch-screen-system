import time

class FPSCalculator:
    def __init__(self) -> None:
        self.p_time = 0;
    
    def get_fps(self):
        c_time = time.time()
        res = None
        if self.p_time != 0:
            res = 1 / (c_time - self.p_time)
        self.p_time = c_time
        return res
