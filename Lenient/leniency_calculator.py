
import math

class LeniencyCalculator(object):

    def __init__(self, K):
        self.K = K

    def calc_leniency(self, temperature):
        return 1. - pow(math.e, -1. * self.K * temperature)
