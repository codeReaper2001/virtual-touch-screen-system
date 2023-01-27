class SmoothenUtil():
    def __init__(self, smoothening:int) -> None:
        self.has_prev = False
        self.px = 0
        self.py = 0
        self.smoothening = smoothening
    
    def get_smooth_val(self, cx:float, cy:float):
        if self.has_prev:
            sx = self.px + (cx - self.px) / self.smoothening
            sy = self.py + (cy - self.py) / self.smoothening
        else:
            sx, sy = cx, cy
            self.has_prev = True
        self.px = sx
        self.py = sy
        return (sx, sy)

        