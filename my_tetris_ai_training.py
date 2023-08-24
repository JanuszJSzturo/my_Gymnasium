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


# HELPER FUNCTIONS
def choose_action(model, observation, single=True):
    """
    Function that takes observations as input, executes a forward pass through model, and outputs a sampled action.
    Arguments:
        model: the network that defines our agent
        observation: observation(s) which is/are fed as input to the model
        single: flag whether we are handling a single observation or batch of observations, provided as an np.array
    Return:
        action: choice of agent action
    """
    # add batch dimension to the observation if only a single example was provided
    main_board = observation["main_board"]
    current_hold_next = to_categorical(np.concatenate((observation["current_piece"], observation["reserved_board"], observation["next_board"]), axis=None), num_classes=8).flatten()

    main_board = np.expand_dims(main_board, axis=0) if single else observation
    current_hold_next = np.expand_dims(current_hold_next, axis=0) if single else observation

    observation = list((main_board, current_hold_next))

    '''TODO: feed the observations through the model to predict the log probabilities of each possible action.'''
    logits_pos, logits_rot = model.predict(observation, verbose=0)

    '''TODO: Choose an action from the categorical distribution defined by the log 
       probabilities of each possible action.'''
    pos = tf.random.categorical(logits_pos, num_samples=1)
    rot = tf.random.categorical(logits_rot, num_samples=1)
    pos = pos.numpy().flatten()
    rot = rot.numpy().flatten()

    return [pos[0], rot[0]] if single else [pos, rot]


def choose_action_2(model, observation, env: my_tetris.TetrisEnv, single=True):
    """
    Function that takes observations as input, executes a forward pass through model, and outputs a sampled action.
    Arguments:
        model: the network that defines our agent
        observation: observation(s) which is/are fed as input to the model
        single: flag whether we are handling a single observation or batch of observations, provided as an np.array
    Return:
        action: choice of agent action
    """

    epsilon = 0.05
    future_states = env.internal_state.get_future_states()
    rng = np.random.default_rng()
    if rng.random() > epsilon:
        # Exploit
        # Get the minumun board energy
        min_energy = min(future_states['board_energies'])
        index = future_states['board_energies'].index(min_energy)
        pos, rot = future_states['positions'][index]
    else:
        pos, rot = random.choice(future_states['positions'])

    pos = np.array(pos, ndmin=1)
    rot = np.array(rot, ndmin=1)

    return [pos[0], rot[0]] if single else [pos, rot]


# Reward function #
def normalize(x):
    """Helper function that normalizes an np.array x """
    x -= np.mean(x)
    x /= np.std(x)
    return x.astype(np.float32)

def discount_rewards(rewards, gamma=0.95):
    """
    Compute normalized, discounted, cumulative rewards (i.e., return)
    Arguments:
        rewards: reward at timesteps in episode
        gamma: discounting factor
    Returns:
        normalized discounted reward
    """
    discounted_rewards = np.zeros_like(rewards)
    R = 0
    for t in reversed(range(0, len(rewards))):
        # update the total discounted reward
        R = R * gamma + rewards[t]
        discounted_rewards[t] = R
    return normalize(discounted_rewards)


# Agent Memory
class Memory:
    def __init__(self):
        self.clear()

    # Resets/restarts the memory buffer
    def clear(self):
        self.observations1 = []
        self.observations2 = []
        self.actions = []
        self.rewards = []

    # Add observations, actions, rewards to memory
    def add_to_memory(self, new_observation, new_action, new_reward):
        main_board = new_observation["main_board"]
        current_hold_next = to_categorical(np.concatenate((new_observation["current_piece"], new_observation["reserved_board"], new_observation["next_board"]),axis=None), num_classes=8).flatten()
        observation = list((main_board, current_hold_next))
        self.observations1.append(main_board)
        self.observations2.append(current_hold_next)
        # Update the list of actions with new action
        self.actions.append(new_action)

        # Update the list of rewards with new reward
        self.rewards.append(new_reward)

    def __len__(self):
        return len(self.actions)


# Loss function
def compute_loss(logits, actions, rewards):
    """
    Arguments:
        logits: network's predictions for actions to take
        actions: the actions the agent took in an episode
        rewards: the rewards the agent received in an episode
    Returns:
        loss
    """
    neg_logprob1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[0], labels=actions[:, 0])
    neg_logprob2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[1], labels=actions[:, 1])

    # Scale the negative log probability by the rewards
    loss = tf.reduce_mean((neg_logprob1+neg_logprob2) * rewards)
    return loss


# Training step (forward and backpropagation)
def train_step(model, loss_function, optimizer, observations1, observations2, actions, discounted_rewards, custom_fwd_fn=None):
    with tf.GradientTape() as tape:
        # Forward propagate through the agent network
        if custom_fwd_fn is not None:
            prediction = custom_fwd_fn([observations1, observations2])
        else:
            prediction = model([observations1, observations2])
            # prediction = model.predict({"main_board": observations1, "current_hold_next_input": observations2})
            #prediction = model.predict(list((observations[:, 0], observations[:, 1])))

        '''TODO: call the compute_loss function to compute the loss'''
        loss = loss_function(prediction, actions, discounted_rewards)

    '''TODO: run backpropagation to minimize the loss using the tape.gradient method. 
             Unlike supervised learning, RL is *extremely* noisy, so you will benefit 
             from additionally clipping your gradients to avoid falling into 
             dangerous local minima. After computing your gradients try also clipping
             by a global normalizer. Try different clipping values, usually clipping 
             between 0.5 and 5 provides reasonable results. '''
    grads = tape.gradient(loss, model.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, 2)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


# env = my_tetris.TetrisEnv(render_mode="human")
env = my_tetris.TetrisEnv()
# env = gym.make("CartPole-v1")
n_observations = env.observation_space
n_actions = env.action_space.n
columns = 10
rotations = 4

# Define the Tetris agent
# Defines a feed-forward neural network
def create_tetris_model_v0():
    shape_main_grid = (1, 20, 10, 1)
    shape_current_hold_next = (1, 64)
    main_grid_input = tf.keras.Input(shape=shape_main_grid[1:], name="main_grid_input")
    a = tf.keras.layers.Conv2D(64, 6, activation="relu", input_shape=shape_main_grid[1:])(main_grid_input)
    a1 = tf.keras.layers.MaxPool2D(pool_size=(15, 5), strides=(1, 1))(a)
    a1 = tf.keras.layers.Flatten()(a1)
    a2 = tf.keras.layers.AvgPool2D(pool_size=(15, 5))(a)
    a2 = tf.keras.layers.Flatten()(a2)

    b = tf.keras.layers.Conv2D(256, 4, activation="relu", input_shape=shape_main_grid[1:])(main_grid_input)
    b1 = tf.keras.layers.MaxPool2D(pool_size=(17, 7), strides=(1, 1))(b)
    b1 = tf.keras.layers.Flatten()(b1)
    b2 = tf.keras.layers.AvgPool2D(pool_size=(17, 7))(b)
    b2 = tf.keras.layers.Flatten()(b2)

    current_hold_next_input = tf.keras.Input(shape=shape_current_hold_next[1:], name="current_hold_next_input")

    x = tf.keras.layers.concatenate([a1, a2, b1, b2, current_hold_next_input])
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    critic_output = tf.keras.layers.Dense(units=n_actions, activation=None)(x)
    position_output = tf.keras.layers.Dense(units=columns, activation="softmax")(x)
    rotation_output = tf.keras.layers.Dense(units=rotations, activation="softmax")(x)

    model = tf.keras.Model(
        inputs=[main_grid_input, current_hold_next_input],
        # outputs=critic_output
        outputs=[position_output, rotation_output]
    )
    model.summary()
    # model = tf.keras.models.Sequential([
    #     # First Dense layer
    #     tf.keras.layers.Dense(units=32, activation='relu'),
    #
    #     # Define the last Dense layer, which will provide the network's output.
    #     tf.keras.layers.Dense(units=n_actions, activation=None)
    # ])
    return model


## Training parameters ##
# Learning rate and optimizer
learning_rate = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate)

# instantiate Tetris agent
tetris_model = create_tetris_model_v0()
memory = Memory()
episodes = 1000
for i_episode in range(episodes):
    # Restart the environment
    observation, info = env.reset()
    memory.clear()
    print(f'Episode {i_episode} out of {episodes}')
    while True:
        # using our observation, choose an action and take it in the environment
        pos, rot = choose_action_2(tetris_model, observation, env)
        # pos, rot = choose_action(tetris_model, observation)
        actions = env.internal_state.movement_planning(pos, rot)
        for action in actions:
            next_observation, reward, done, truncated, info = env.step(action)
        # add to memory
        memory.add_to_memory(observation, [pos, rot], reward)

        # is the episode over? did you crash or do so well that you're done?
        if done:
            # determine total reward and keep a record of this
            total_reward = sum(memory.rewards)
            # initiate training - remember we don't know anything about how the
            # agent is doing until it has crashed!
            train_step(tetris_model, compute_loss, optimizer,
                       observations1=np.array(memory.observations1),
                       observations2=np.array(memory.observations2),
                       actions=np.array(memory.actions),
                       discounted_rewards=discount_rewards(memory.rewards))
            print(memory.rewards)

            # reset the memory
            memory.clear()
            break
        # update our observations
        observation = next_observation

tetris_model.save('tetris_model_v0_1000_ep.keras')
