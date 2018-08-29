
from Env.cmotp_IL import CMOTP
from Lenient.lenient_DQNAgent import LenientDQNAgent
import numpy as np

TRAIN_EPISODES = 5000

TRAIN = True
TEST_NUM = 10


if __name__ == '__main__':
    env = CMOTP()
    agent1 = LenientDQNAgent(env, [256, 256], 'LenientAgent1',
                             learning_rate=1e-4, replay_memory_size=100000,
                             targetnet_update_freq=5000)

    agent2 = LenientDQNAgent(env, [256, 256], 'LenientAgent1',
                             learning_rate=1e-4, replay_memory_size=100000,
                             targetnet_update_freq=5000)


    if TRAIN:

        for i in range(TRAIN_EPISODES):
            state1, state2 = env.reset()
            episode_len = 0
            episode1, episode2 = [], []

            while True:
                action1 = agent1.choose_action(state1)
                action2 = agent2.choose_action(state2)
                next_state, reward, done, _ = env.step([action1, action2])
                next_state1, next_state2 = next_state
                reward1, reward2 = reward
                done1, done2 = done
                leniency1 = agent1.leniency_calculator.calc_leniency(agent1.temp_recorder.get_state_action_temp(state1, action1))
                leniency2 = agent2.leniency_calculator.calc_leniency(agent2.temp_recorder.get_state_action_temp(state2, action2))

