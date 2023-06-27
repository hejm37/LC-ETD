from Algorithms.LCETD import LCETD


class ScaledETDLB(LCETD):
    def __init__(self, task, **kwargs):
        super().__init__(task, **kwargs)
        self.type = 'left'
