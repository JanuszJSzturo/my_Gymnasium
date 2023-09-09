import pygame
from TetrisState import TetrisState, Action
import numpy as np
import TestTetrisStates as tts


class MyGUI:
    def __init__(self):
        self.state = TetrisState()
        self.cell_size = 32
        self.height = 23
        self.width = 25
        self.window_size = np.array((self.width * self.cell_size, self.height * self.cell_size))
        self.game_over = False
        self.quit = False
        self.action = None
        self.clock = pygame.time.Clock()
        self.next = True
        self.copied_state = None

        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode(self.window_size)
        self.canvas = pygame.Surface(self.window_size)

        self.state.load_state(tts.GRID_STATE_6)

    def run(self):
        while not self.game_over and not self.quit:
            self.process_input()
            self.update_human()
            if self.next:
                self.update()
            self.render()

    def process_input(self):
        self.action = None
        self.next = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.quit = True
                elif event.key == pygame.K_a:
                    self.action = Action.LEFT
                elif event.key == pygame.K_d:
                    self.action = Action.RIGHT
                elif event.key == pygame.K_s:
                    self.action = Action.DOWN
                elif event.key == pygame.K_w:
                    self.action = Action.DROP
                elif event.key == pygame.K_j:
                    self.action = Action.ROT_LEFT
                elif event.key == pygame.K_k:
                    self.action = Action.ROT_RIGHT
                elif event.key == pygame.K_r:
                    self.action = Action.RESERVE
                elif event.key == pygame.K_SPACE:
                    self.next = True
                elif event.key == pygame.K_c:
                    self.show_future_states()
                elif event.key == pygame.K_z:
                    self.action = Action.SOFT

    def update(self):
        x_pos = np.random.choice(range(10))
        rot_state = np.random.choice(range(4))
        print(f'Start position: {self.state.current_tetr._x}, start rotation: {self.state.current_tetr._rotation_state}')
        print(f'End Position: {x_pos}, end rotation: {rot_state}')
        movements = self.state.movement_planning(x_pos, rot_state)
        print(movements)
        for move in movements:
            self.game_over, _, _ = self.state.update(move.value)

    def update_human(self):
        self.game_over, _, _ = self.state.update(action=self.action)


    def show_future_states(self):
        initial_state = self.state.save_state()
        ghost_state = TetrisState()
        ghost_state.load_state(initial_state)

        future_states = ghost_state.get_future_states()
        print(f'---Total positions: {len(future_states["movements"])}---')
        for i, movements in enumerate(future_states["movements"]):
            print(f'  Position: {i}')
            for move in movements:
                over, piece_locked, lines_cleared = ghost_state.update(move.value)
            print(f'    col: {ghost_state.played_tetr[-1].left}, rot: {ghost_state.played_tetr[-1]._rotation_state}')
            self.canvas.fill((0, 0, 0))
            self.canvas = ghost_state.render_frame(self.canvas, self.cell_size)
            self.window.blit(self.canvas, self.canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            pygame.time.delay(500)
            ghost_state.load_state(initial_state)



    def render(self):
        self.canvas.fill((0, 0, 0))

        self.canvas = self.state.render_frame(self.canvas, self.cell_size)
        self.window.blit(self.canvas, self.canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()


if __name__ == '__main__':
    game = MyGUI()
    game.run()
    game = MyGUI()
    pygame.quit()

