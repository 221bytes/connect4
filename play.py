# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
import tensorflow as tf
import gym_connect4

if __name__ == "__main__":
    env = gym.make('connect4-v0')
    # env = gym.make('CartPole-v1')
    # state_size = env.observation_space.shape[0]
    # action_size = env.action_space.n
    # agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 32

    # for e in range(EPISODES):
    #     state = env.reset()
    #     state = np.reshape(state, [1, state_size])

    for time in range(500):
        env.render()
        sec = input('Let us wait for user input. Let me know how many seconds to sleep now.\n')
        env.step(int(sec), 1)

        # action = agent.act(state)
        # next_state, reward, done, _ = env.step("0")
        # reward = reward if not done else -10
        # next_state = np.reshape(next_state, [1, state_size])
        # # agent.remember(state, action, reward, next_state, done)
        # state = next_state
        # if done:
        #     # print("episode: {}/{}, score: {}, e: {:.2}"
        #     #         .format(e, EPISODES, time, agent.epsilon))
        #     print("done")
        #     break
        # if len(agent.memory) > batch_size:
        #     agent.replay(batch_size)
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-dqn.h5")

    env.close()