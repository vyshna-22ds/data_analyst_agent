import time
class Deadline:
    def __init__(self, budget_s: float = 170.0):
        self.start = time.monotonic(); self.budget = budget_s
    @property
    def elapsed(self): return time.monotonic() - self.start
    @property
    def remaining(self): return max(0.0, self.budget - self.elapsed)
    def nearly_out(self, threshold: float = 10.0): return self.remaining < threshold
