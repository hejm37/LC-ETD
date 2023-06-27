from Algorithms.LCETD import LCETD


class FullISTD(LCETD):
    def __init__(self, task, **kwargs):
        super().__init__(task, **kwargs)
        self.beta = 1
