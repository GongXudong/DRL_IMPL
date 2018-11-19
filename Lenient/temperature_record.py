
import numpy as np
import math

class Temp_record(object):

    def __init__(self, shape: tuple, beta_len, scale=1, beta_p=-0.01, beta_d=0.95, v=1., u=0.999):
        """

        :param shape: (s, a)
        :param beta_len:
        :param beta_p:
        :param beta_d:
        :param v:
        :param u:
        """
        self.temps = np.ones(shape, dtype=np.float32)
        self.betas = np.zeros(beta_len * scale)
        self.beta_len = beta_len
        self.scale = scale
        self.beta_p = beta_p
        self.beta_d = beta_d
        self._init_beta()
        self.v = v
        self.u = u

    def _init_beta(self):
        tmp = self.beta_p * 1.
        for i in range(self.beta_len):
            for j in range(self.scale):
                self.betas[i * self.scale + j] = pow(math.e, tmp)
            tmp *= self.beta_d
        self.beta_len *= self.scale

    def show_beta(self):
        print(self.betas)

    def convert_to_tuple(self, s: np.ndarray, a: int) -> tuple:
        return tuple(np.int32(s)) + (a, )

    def convert_to_string(self, s: np.ndarray, a: int) -> str:
        return self.convert_to_tuple(s, a).__str__()

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
        for s, a in episode[::-1]:
            s_a_tuple = self.convert_to_tuple(s, a)
            tmp = self.temps[s_a_tuple] * (1.0 if n >= self.beta_len else self.betas[n])
            if tmp < self.v:
                self.temps[s_a_tuple] = tmp
            else:
                self.temps[s_a_tuple] = self.v
            n += 1

        # decay v
        self.v *= self.u

    def save(self, savedir='.'):
        average_temp = np.zeros(self.temps.shape[0:2])
        for i in range(0, self.temps.shape[0]):
            for j in range(0, self.temps.shape[1]):
                average_temp[i][j] = np.mean(self.temps[(i, j)])
        np.save(savedir + '/ave_temp.npy', average_temp)


class Temp_record_with_dict(object):
    def __init__(self, beta_len, scale=1, beta_p=-0.01, beta_d=0.95, v=1., u=0.999):
        """

        :param shape: (s, a)
        :param beta_len:
        :param beta_p:
        :param beta_d:
        :param v:
        :param u:
        """
        self.temps = {}
        self.betas = np.zeros(beta_len * scale)
        self.beta_len = beta_len
        self.scale = scale
        self.beta_p = beta_p
        self.beta_d = beta_d
        self._init_beta()
        self.v = v
        self.u = u

    def _init_beta(self):
        tmp = self.beta_p * 1.
        for i in range(self.beta_len):
            for j in range(self.scale):
                self.betas[i * self.scale + j] = pow(math.e, tmp)
            tmp *= self.beta_d
        self.beta_len *= self.scale

    def convert_to_tuple(self, s: np.ndarray, a: int) -> tuple:
        return tuple(np.int32(s)) + (a, )

    def get_state_action_temp(self, s: np.ndarray, a: int):
        s_a_tuple = self.convert_to_tuple(s, a)
        if not self.temps.__contains__(s_a_tuple):
            self.temps[s_a_tuple] = 1.
        return self.temps[s_a_tuple]

    def get_state_temp(self, s: np.ndarray, actions=range(5)):
        tmp = []
        for act in actions:
            s_a_temp = self.get_state_action_temp(s, act)
            tmp.append(s_a_temp)
        return np.mean(tmp)

    def decay_temp(self, episode):
        """

        :param episode: list[tuple[np.ndarray, int]]
        :return:
        """
        # decay the episode
        n = 0
        for s, a in episode[::-1]:
            s_a_tuple = self.convert_to_tuple(s, a)
            s_a_temp = self.get_state_action_temp(s, a)
            tmp = s_a_temp * (1.0 if n >= self.beta_len else self.betas[n])
            if tmp < self.v:
                self.temps[s_a_tuple] = tmp
            else:
                self.temps[s_a_tuple] = self.v
            n += 1

        # decay v
        self.v *= self.u

    def get_ave_temp(self):
        return np.mean(list(self.temps.values()))

    def get_temp_len(self):
        return len(self.temps)

    def show_temp(self, big=True, narrow=False):
        if big and not narrow:
            a = [[11, 0, 1, 11, 10, 1],
                 [11, 4, 1, 11, 6, 1],
                 [7, 4, 2, 7, 6, 3],
                 [7, 6, 3, 7, 4, 2],
                 [5, 4, 2, 5, 6, 3],
                 [5, 6, 3, 5, 4, 2]]
            # a = [[15, 0, 1, 15, 14, 1],
            #      [15, 6, 1, 15, 8, 1],
            #      [11, 6, 2, 11, 8, 3],
            #      [11, 8, 3, 11, 6, 2],
            #      [8, 6, 2, 8, 8, 3],
            #      [8, 8, 3, 8, 6, 2],
            #      [3, 6, 2, 3, 8, 3],
            #      [3, 8, 3, 3, 6, 2]]
        elif big and narrow:
            a = [[11, 0, 1, 11, 10, 1],
                 [11, 4, 1, 11, 6, 1],
                 [8, 10, 3, 8, 8, 2],
                 [8, 8, 2, 8, 10, 3],
                 [3, 7, 3, 3, 5, 2],
                 [3, 5, 2, 3, 7, 3]]
        else:
            a = [[2, 2, 2, 2, 4, 3],
                 [2, 4, 3, 2, 2, 2],
                 [0, 4, 2, 0, 6, 3],
                 [0, 6, 3, 0, 4, 2],
                 [0, 0, 2, 0, 2, 3],
                 [0, 2, 3, 0, 0, 2]]
        res = []
        for s in a:
            res.append(self.get_state_temp(np.array(s)))
        print(res)
        return (self.get_state_temp(np.array([7, 4, 2, 7, 6, 3])),
                self.get_state_temp(np.array([7, 6, 3, 7, 4, 2])))


if __name__ == '__main__':
    # tr = Temp_record((2, 2, 2), 500)
    # aa = tr.convert_to_tuple(np.array([1,2,3,4]), 2)
    # print(aa.__str__())
    # for i in range(len(tr.betas)):
    #     print(tr.betas[i])
    tr = Temp_record_with_dict(3, beta_p=-0.5, beta_d=0.8)
    episode = [(np.array([0, 0]), 0), (np.array([0, 1]), 1), (np.array([0, 0]), 0)]
    tr.decay_temp(episode)
    print(tr.temps)
    print(tr.get_state_temp(np.array([0, 0])))
    print(tr.get_ave_temp())
    print(tr.temps)





