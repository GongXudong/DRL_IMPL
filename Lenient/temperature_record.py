
import numpy as np
import math

class Temp_record(object):

    def __init__(self, shape: tuple, beta_len, beta_p=-0.01, beta_d=0.95):
        self.temps = np.ones(shape, dtype=np.float32)
        self.betas = np.zeros(beta_len)
        self.beta_len = beta_len
        self.beta_p = beta_p
        self.beta_d = beta_d
        self._init_beta()


    def _init_beta(self):
        tmp = self.beta_p * 1.
        for i in range(self.beta_len):
            self.betas[i] = pow(math.e, tmp)
            tmp *= self.beta_d

    def show_beta(self):
        print(self.betas)

    def convert_to_tuple(self, s: np.ndarray, a: int) -> tuple:
        return tuple(s) + (a, )



if __name__ == '__main__':
    obj = Temp_record((1, 2), 1000)
    print(obj.betas[600])
    print(obj.convert_to_tuple(np.array([1, 2, 3]), 0))



