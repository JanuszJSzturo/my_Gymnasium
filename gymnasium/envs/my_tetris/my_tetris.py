import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from gymnasium.utils.play import play
from gymnasium.envs.my_tetris.TetrisState import TetrisState


class TetrisEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 24}

    def __init__(self, render_mode=None, size=32):
        pygame.init()
        self.internal_state = TetrisState()
        self.cell_size = size  # The size of a square cell
        self.height = 23  # Height in cells of PyGame window
        self.width = 25  # Width in cells of PyGame window
        self.window_size = np.array((self.width * self.cell_size, self.height * self.cell_size))  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).

        self.observation_space = spaces.Dict(
            {
                "main_board": spaces.MultiBinary([20, 10]),
                "next_board": spaces.MultiDiscrete([7, 7, 7, 7, 7, 7]),
                "reserved_board": spaces.Discrete(7),
                "current_piece": spaces.Discrete(7)
            }
        )

        # We have 7 actions, corresponding to "left", "right", "down", "rotate_left", "rotate_right", "drop", "hold"
        self.action_space = spaces.Discrete(7)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"main_board": self.internal_state.get_board(),
                "next_board": self.internal_state.get_next_tetrominoes(),
                "reserved_board": self.internal_state.get_reserved(),
                "current_piece": self.internal_state.get_current_tetromino()}

    def _get_info(self):
        return {"total_lines_cleared": self.internal_state.get_total_lines_cleared(),
                "score": self.internal_state.get_score(),
                "num_actions": self.internal_state.get_num_actions()}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.internal_state.reset()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # TODO define "terminated", "reward", "observation" and "info"
        # Process the step
        terminated, piece_locked, df_score = self.internal_state.update(action)

        # Get the observation and info of new state
        observation = self._get_obs()
        info = self._get_info()

        # Calculate the reward
        reward = df_score
        if terminated:
            reward = self.internal_state.score

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(self.window_size)
        canvas.fill((0, 0, 0))

        canvas = self.internal_state.render_frame(canvas, self.cell_size)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


if __name__ == "__main__":
    env = TetrisEnv(render_mode="rgb_array")
    mapping = {(pygame.K_RIGHT,): 1,
               (pygame.K_DOWN,): 2,
               (pygame.K_LEFT,): 0,
               (pygame.K_a,): 3,
               (pygame.K_s,): 4,
               (pygame.K_UP,): 5,
               (pygame.K_r,): 6}
    play(env, keys_to_action=mapping)
