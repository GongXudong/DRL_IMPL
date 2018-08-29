
import numpy as np
import math

class Temp_record(object):

    def __init__(self, shape: tuple, beta_len, beta_p=-0.01, beta_d=0.95, v=1., u=0.999):
        """

        :param shape: (s, a)
        :param beta_len: 
        :param beta_p:
        :param beta_d:
        :param v:
        :param u:
        """
        self.temps = np.ones(shape, dtype=np.float32)
        self.betas = np.zeros(beta_len)
        self.beta_len = beta_len
        self.beta_p = beta_p
        self.beta_d = beta_d
        self._init_beta()
        self.v = v
        self.u = u

    def _init_beta(self):
        tmp = self.beta_p * 1.
        for i in range(self.beta_len):
            self.betas[i] = pow(math.e, tmp)
            tmp *= self.beta_d

    def show_beta(self):
        print(self.betas)

    def convert_to_tuple(self, s: np.ndarray, a: int) -> tuple:
        return tuple(np.int32(s)) + (a, )

    def get_state_action_temp(self, s: np.ndarray, a: int):
        s_a_tuple = self.convert_to_tuple(s, a)
        return self.temps[s_a_tuple]

    def get_state_temp(self, s: np.ndarray):
        cur_s = tuple(np.int32(s))
        return np.mean(self.temps[cur_s])

    def decay_temp(self, episode):
        """

        :param episode: list[tuple[np.ndarray, int]]
        :return:
        """
        # decay the episode
        n = 0
        for s, a in episode:
            s_a_tuple = self.convert_to_tuple(s, a)
            tmp = self.temps[s_a_tuple] * (1.0 if n >= self.beta_len else self.betas[n])
            if tmp < self.v:
                self.temps[s_a_tuple] = tmp
            else:
                self.temps[s_a_tuple] = self.v
            n += 1

        # decay v
        self.v *= self.u


if __name__ == '__main__':
    tr = Temp_record((2, 2, 2), 3, beta_p=-0.5, beta_d=0.8)
    episode = [(np.array([0, 0]), 0), (np.array([0, 1]), 1), (np.array([0, 0]), 0)]
    tr.decay_temp(episode)
    print(tr.temps)





