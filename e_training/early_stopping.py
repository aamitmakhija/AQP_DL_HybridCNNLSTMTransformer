class EarlyStopper:
    def __init__(self, patience: int, mode: str="min"):
        self.patience = int(patience)
        self.mode = mode
        self.best = None
        self.bad = 0
    def step(self, value: float) -> bool:
        if self.best is None:
            self.best = value; self.bad = 0; return False
        better = (value < self.best) if self.mode=="min" else (value > self.best)
        if better:
            self.best = value; self.bad = 0; return False
        self.bad += 1
        return self.bad > self.patience