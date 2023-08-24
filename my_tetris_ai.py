import random

import tensorflow as tf
import numpy as np
import base64, io, os, time, gymnasium
from gymnasium.envs.my_tetris import my_tetris
import functools
import time
from keras.utils import to_categorical
# import tensorflow_probability as tfp
# import mitdeeplearning as mdl


loaded_model = tf.keras.models.load_model("tetris_model_v0_1000_ep.keras")


env = my_tetris.TetrisEnv(render_mode="human")

games = 10
loaded_model.summary()

for i_game in range(games):
    observation, info = env.reset()
    done = False
    print(f'Game {i_game} out of {games}')
    while not done:
        main_board = observation["main_board"]
        current_hold_next = to_categorical(
            np.concatenate((observation["current_piece"], observation["reserved_board"], observation["next_board"]),
                           axis=None), num_classes=8).flatten()

        main_board = np.expand_dims(main_board, axis=0)
        current_hold_next = np.expand_dims(current_hold_next, axis=0)

        observation = list((main_board, current_hold_next))
        logits_pos, logits_rot = loaded_model.predict(observation)
        print(logits_pos)

        pos = tf.random.categorical(logits_pos, num_samples=1)
        rot = tf.random.categorical(logits_rot, num_samples=1)
        pos = pos.numpy().flatten()
        rot = rot.numpy().flatten()
        print(pos)

        actions = env.internal_state.movement_planning(int(pos), int(rot))
        for action in actions:
            next_observation, reward, done, truncated, info = env.step(action)
        observation = next_observation

