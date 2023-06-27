from Algorithms.TD import TD
from Algorithms.ETDLB import ETDLB
from Algorithms.ETD import ETD
from Algorithms.LCETD import LCETD, LCETD1, LCETD2, LCETD3
from Algorithms.ScaledETDLB import ScaledETDLB
from Algorithms.FullISTD import FullISTD

alg_dict = {'TD': TD, 'ETD': ETD, 'ETDLB': ETDLB, 'LCETD': LCETD,
            'ScaledETDLB': ScaledETDLB, 'LCETD': LCETD, 'FullISTD': FullISTD,
            'LCETD1': LCETD1, 'LCETD2': LCETD2, 'LCETD3': LCETD3}
