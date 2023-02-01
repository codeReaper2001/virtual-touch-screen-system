class SmoothenUtil():
    def __init__(self, smoothening:int) -> None:
        self.reset()
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

    def get_px_py(self):
        return self.px, self.py

    def reset(self):
        self.has_prev = False
        self.px = 0.0
        self.py = 0.0
        