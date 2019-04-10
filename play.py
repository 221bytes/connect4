# -*- coding: utf-8 -*-
import random
import gym
import time
import numpy as np
from collections import deque
import tensorflow as tf
import gym_connect4
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
sess.as_default()
# if __name__ == "__main__":
#     env = gym.make('connect4-v0')
# #     # env = gym.make('CartPole-v1')
# #     # state_size = env.observation_space.shape[0]
# #     # action_size = env.action_space.n
# #     # agent = DQNAgent(state_size, action_size)
# #     # agent.load("./save/cartpole-dqn.h5")
# #     done = False
# #     batch_size = 32

# #     # for e in range(EPISODES):
# #     #     state = env.reset()
# #     #     state = np.reshape(state, [1, state_size])
# #     for i_episode in range(100000):
# #         observation = env.reset()
# #         for t in range(100):
# #             # env.render()
# #             action = env.action_space.sample()
# #             observation, reward, done, info = env.step(action,1)
# #             if done:
# #                 # print("Episode finished after {} timesteps".format(t+1))
# #                 print(i_episode, reward)
# #                 # if i_episode % 1000 == 0:
# #                     # print(i_episode, reward)
# #                 break
# #     env.close()
#     for time in range(500):
#         env.render()
#         sec = input('Let us wait for user input. Let me know how many seconds to sleep now.\n')
#         _, reward, done, _ = env.step(int(sec), 1)
#         print(reward, done)
#         # next_state, reward, done, _ = env.step(0,0)

#         # action = agent.act(state)
#         # next_state, reward, done, _ = env.step("0")
#         # reward = reward if not done else -10
#         # next_state = np.reshape(next_state, [1, state_size])
#         # # agent.remember(state, action, reward, next_state, done)
#         # state = next_state
#         # if done:
#         #     # print("episode: {}/{}, score: {}, e: {:.2}"
#         #     #         .format(e, EPISODES, time, agent.epsilon))
#         #     print("done")
#         #     break
#         # if len(agent.memory) > batch_size:
#         #     agent.replay(batch_size)
#         # if e % 10 == 0:
#         #     agent.save("./save/cartpole-dqn.h5")

#     env.close()

# -*- coding: utf-8 -*-

EPISODES = 100000


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 0.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # # Neural Net for Deep-Q learning Model
        # model = tf.keras.Sequential()

        # model.add(tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'))
        # model.add(tf.keras.layers.Dense(24, activation='relu'))
        # model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        # model.compile(loss='mse',
        #               optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        keras = tf.keras
        layers = keras.layers
        model = keras.Sequential()
        model.add(layers.Conv2D(32, kernel_size=(2, 2),
                                activation='relu',
                                input_shape=(6, 7, 1)))  # connect 4 board
        # model.add(Conv2D(64, (3, 3), activation='relu'))
        # model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        # model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(32, (2, 2), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(7,  # columns of the board
                               activation='softmax'))

        # # hidden layer takes a pre-processed frame as input, and has 200 units
        # model.add(layers.Dense(units=200,input_dim=6*7, activation='relu', kernel_initializer='glorot_uniform'))

        # # output layer
        # model.add(layers.Dense(units=7, activation='softmax', kernel_initializer='RandomNormal'))

        model.compile(loss='mse',
                      optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))

        model.summary()
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = np.reshape(state, (1, 6, 7, 1))
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            state = np.reshape(state, (1, 6, 7, 1))

            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.memory.clear()

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = gym.make('connect4-v0')
    # env = gym.make('CartPole-v1')
    state_size = [6, 7]  # env.observation_space.shape[0]
    action_size = 7  # env.action_space.n
    agent_1 = DQNAgent(state_size, action_size)
    agent_2 = DQNAgent(state_size, action_size)
    agent_1.load("./save/agent_1_connect4-dqn.h5")
    agent_2.load("./save/agent_2_connect4-dqn.h5")

    done = False
    batch_size = 32

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, (1, 6, 7, 1))
        reward_1 = 0
        reward_2 = 0
        for t in range(21):
            env.render()

            sec = input('Let us wait for user input. Let me know how many seconds to sleep now.\n')
            action_1 = int(sec)
            
            # action_1 = agent_1.act(state)
            next_state, reward_1, done_1, _ = env.step(action_1, 1, reward_1)
            next_state = np.reshape(next_state, (1, 6, 7, 1))
            # agent_1.remember(state, action_1, reward_1, next_state, done)

            # state = next_state

            sec = input('Let us wait for user input. Let me know how many seconds to sleep now.\n')
            env.render()

            action_2 = agent_2.act(state)
            next_state, reward_2, done_2, _ = env.step(action_2, 2, reward_2)
            next_state = np.reshape(next_state, (1, 6, 7, 1))
            if done_1:
                reward_2 += -10.0
            agent_2.remember(state, action_2, reward_2, next_state, done)

            state = next_state

            # sec = input('Let us wait for user input. Let me know how many seconds to sleep now.\n')

            if done_1 or done_2:
                print("episode: {}/{}, score: agent_1 : {} agent_2 : {} , e1: {:.2} e2: {:.2}"
                      .format(e, EPISODES, reward_1, reward_2, agent_1.epsilon, agent_2.epsilon))
                # env.render()
                # time.sleep(2)
                break
        if len(agent_1.memory) > batch_size:
            agent_1.replay(batch_size)
            agent_2.replay(batch_size)
        # if e % 10 == 0:
        #     agent_1.save("./save/agent_1_connect4-dqn.h5")
        #     agent_2.save("./save/agent_2_connect4-dqn.h5")
