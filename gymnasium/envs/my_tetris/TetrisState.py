from enum import Enum
import numpy as np
from gymnasium.envs.my_tetris.Tetromino import Tetromino, BlockID, DotBlock
import time
import pygame
import random

BOARD_WIDTH = 10
BOARD_HEIGHT = 20

NUM_NEXT_TETR = 6

DISTANCE_SCORE = 1

LEFT = np.array([-1, 0])
RIGHT = np.array([1, 0])
DOWN = np.array([0, 1])

ROT_RIGHT = -1
ROT_LEFT = 1

NO_GRAVITY = np.inf
DEFAULT_GRAVITY = 1000*10**6

LOCK_DELAY_TIME = 500*10**6

# Look-up tables for wall kicks for pieces J, L, T, S, Z
# NOTE: the sign convention is the same as stated in https://tetris.fandom.com/wiki/SRS:
# positive x is rightwards and positive y is upwards.
WALL_KICKS_JLTSZ = ([[[(-1, 0), (-1, 1), (0, -2), (-1, -2)], [(1, 0), (1, 1), (0, -2), (1, -2)]],
                     [[(1, 0), (1, -1), (0, 2), (1, 2)], [(1, 0), (1, -1), (0, 2), (1, 2)]],
                     [[(1, 0), (1, 1), (0, -2), (1, -2)], [(-1, 0), (-1, 1), (0, -2), (-1, -2)]],
                     [[(-1, 0), (-1, -1), (0, 2), (-1, 2)], [(-1, 0), (-1, -1), (0, 2), (-1, 2)]]])

# Look-up tables for wall kicks for piece I
WALL_KICKS_I = ([[[(-2, 0), (1, 0), (-2, -1), (1, 2)], [(-1, 0), (2, 0), (-1, 2), (2, -1)]],
                 [[(-1, 0), (2, 0), (-1, 2), (2, -1)], [(2, 0), (-1, 0), (2, 1), (-1, -2)]],
                 [[(2, 0), (-1, 0), (2, 1), (-1, -2)], [(1, 0), (-2, 0), (1, -2), (-2, 1)]],
                 [[(1, 0), (-2, 0), (1, -2), (-2, 1)], [(-2, 0), (1, 0), (-2, -1), (1, 2)]]])


class Action(Enum):
    LEFT = 0
    RIGHT = 1
    DOWN = 2
    ROT_LEFT = 3
    ROT_RIGHT = 4
    DROP = 5
    RESERVE = 6


# TODO: Comment the code
# TODO: test the game mechanics
# TODO: Define reward function
# TODO: Define observations
# TODO: implement score system
# TODO: configure automatic block falling if possible and speed up according lines cleared
# TODO: tetrominnoes spawn 2 blocks lower if can
# TODO: configure tetris level
class TetrisState:
    """
    Tetris game representation and interaction
    Board: numpy array of size (20, 10) of int.
        - 1 -> occupied cell
        - 0 -> free cell
    Current, reserved -> Tetromino object
    Played, next -> list of Tetromino objects
    """
    def __init__(self):
        """
        Initialize a tetris game state
        Clean the board
        Clean reserved tetromino
        Clean played tetrominoes
        Initializes current tetromino with a random tetromino
        Populate future tetrominoes with 6 random tetromino
        """
        self.board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int)

        self.played_tetr = []
        self.reserved_tetr = None
        self.next_tetr = []
        self.can_reserve = True
        self.time = time.monotonic_ns()
        self.starting_time = time.monotonic_ns()
        self.time_step = DEFAULT_GRAVITY
        self.lock_delay_started = False
        self.lock_time = 0
        self.game_ticks = 0

        self.randomizer_7_bag = list(BlockID)
        random.shuffle(self.randomizer_7_bag)
        self.current_tetr = Tetromino.make(self.randomizer_7_bag.pop())

        self.lines = 0
        self.score = 0
        self.pieces_placed = 0
        self.combo = 0
        self.level = 1

        self.actions_done = np.zeros(100)
        self.n_actions = 0
        self.last_piece_num_actions = 0

        for n in range(NUM_NEXT_TETR):
            self.next_tetr.append(Tetromino.make(self.randomizer_7_bag.pop()))

    def reset(self):
        self.__init__()

    def update(self, action):
        self.game_ticks += 1
        over = False
        piece_locked = False
        df_score = 0
        collided = True
        dt = time.monotonic_ns()-self.time
        try:
            action = Action(action)
        except ValueError:
            action = None

        if action == Action.LEFT:
            collided = self._move(self.current_tetr, LEFT)
        elif action == Action.RIGHT:
            collided = self._move(self.current_tetr, RIGHT)
        elif action == Action.DOWN:
            collided = self._move(self.current_tetr, DOWN)
            if not collided:
                self.score += DISTANCE_SCORE
        elif action == Action.ROT_LEFT:
            collided = self._rotate(self.current_tetr, ROT_LEFT)
        elif action == Action.ROT_RIGHT:
            collided = self._rotate(self.current_tetr, ROT_RIGHT)
        elif action == Action.DROP:
            score = self._drop(self.current_tetr)
            self.score += score
            piece_locked = True
        elif action == Action.RESERVE:
            self._reserve()


        # TODO: how should be the priority in gravity over actions?
        if dt >= self.time_step:
            _ = self._move(self.current_tetr, DOWN)
            self.time = time.monotonic_ns()

        self._update_board()

        if not collided and self.lock_delay_started:
            self.lock_delay_started = False
            self.time = time.monotonic_ns()
        # After the movement is resolved, check if piece is at bottom, i.e. it has a piece below or the ground
        if not self.lock_delay_started and self.current_tetr.status == 'playing':
            bottom_reached_locked = self._move(self.current_tetr, DOWN)
            if bottom_reached_locked:
                self.lock_delay_started = True
                self.lock_time = time.monotonic_ns()
            else:
                self._move(self.current_tetr, np.array([0, -1]))
        else:
            dt_lock = time.monotonic_ns() - self.lock_time
            if dt_lock > LOCK_DELAY_TIME:
                self.current_tetr.status = 'locked'
                piece_locked = True
                self.lock_delay_started = False

        if action is not None:
            self.actions_done[self.n_actions] = int(action.value)
            self.n_actions += 1

        if self.current_tetr.status == 'locked':
            # If current tetromino is locked we check line, spawn a new tetromino and check if it is game over
            self.pieces_placed += 1
            self.played_tetr.append(self.current_tetr)
            self._update_board()
            color = self.current_tetr.color
            factor = 0.6
            self.current_tetr.color = (int(color[0]*factor), int(color[1]*factor), int(color[2]*factor))
            lines_cleared = self._check_line()
            score = 0
            if self._t_spin():
                match lines_cleared:
                    case 0:
                        score = 400
                    case 1:
                        score = 800
                    case 2:
                        score = 1200
                    case 3:
                        score = 1600
            else:
                match lines_cleared:
                    case 1:
                        score = 100
                    case 2:
                        score = 300
                    case 3:
                        score = 400
                    case 4:
                        score = 800

            if lines_cleared > 0:
                self.combo += 1
                self.lines += lines_cleared
            else:
                self.combo = 0
            df_score = (1 + self.combo * 0.5) * (score * self.level)
            self.score += df_score
            over = self._spawn_tetromino()
            self.can_reserve = True
            self._update_board()

        if self.lines >= 100:
            print("Reached 100 lines cleared")
            over = True
        if over:
            print(self.score, self.lines, self.pieces_placed)
        return over, piece_locked, df_score

    def load(self, board_state: dict):
        self.reset()
        rows, cols = np.where(board_state["board"] == 1)
        for r, c in zip(rows, cols):
            dot = DotBlock()
            x, y, _ = dot.get_state()
            dot.move((c-1-x, r-1-y))
            self.played_tetr.append(dot)

        if board_state["current"]:
            self.current_tetr = Tetromino.make(board_state["current"])

        next_tetr = board_state["next"][:NUM_NEXT_TETR]
        for i, tetr in enumerate(next_tetr):
            self.next_tetr[i] = Tetromino.make(tetr)

        self._update_board()

    def _update_board(self):
        temp_board = np.zeros((21, 10), dtype=int)
        # current_struct = np.where(self.current_tetr.struct)
        # current_struct = current_struct[0] + self.current_tetr._y, current_struct[1] + self.current_tetr._x
        # temp_board[current_struct] += 1
        for tetr in self.played_tetr:
            temp_struct = np.where(tetr.struct)
            temp_struct = temp_struct[0] + tetr._y, temp_struct[1] + tetr._x
            temp_board[temp_struct] += 1
        self.board = temp_board[:20]

    def _spawn_tetromino(self):
        self.actions_done = np.zeros(100)
        self.last_piece_num_actions = self.n_actions
        self.n_actions = 0
        new_current = self.next_tetr.pop(0).name
        if not self.randomizer_7_bag:
            self.randomizer_7_bag = list(BlockID)
            random.shuffle(self.randomizer_7_bag)

        self.next_tetr.append(Tetromino.make(self.randomizer_7_bag.pop()))
        self.current_tetr = Tetromino.make(new_current)
        terminated = self._collision(self.current_tetr)

        return terminated

    def _reserve(self):
        if self.reserved_tetr is None:
            self.reserved_tetr = Tetromino.make(self.current_tetr.name)
            self._spawn_tetromino()
        elif self.can_reserve:
            current_tetr_name = self.current_tetr.name
            self.current_tetr = Tetromino.make(self.reserved_tetr.name)
            self.reserved_tetr = Tetromino.make(current_tetr_name)
        self.can_reserve = False

    def _check_line(self):
        """
        The idea for checking and updating the grid when a line is complete is as follows:
        1. Indentify the rows where the line is complete (we do it if a row sums 10)
        2. For every row, we check every played tetromino if it is affected
            2.1. Get np.where(tetr.struct) and add the position of the tetr
            2.2. If any row of the tetr is equal to the row affected we change the 1's to 0's in the tetr struct
        3. We need to move down each tetromino accordingly
        """

        # List of tetrominoes that have to be removed when line is cleared
        tetr_to_remove = []

        # Check full rows
        board = self.get_board()
        affected_rows = np.where(board.sum(axis=1) == BOARD_WIDTH)[0]
        # Do nothing if there are no full lines
        if affected_rows.size != 0:
            for row in affected_rows:
                # Get all Tetrominoes that are affected by the row clear
                affected_tetr = TetrisState._tetr_in_row(self.played_tetr, row)
                for tetr in affected_tetr:
                    # Delete the row of the struct affected
                    tetr.struct = np.delete(tetr.struct, row-tetr._y, 0)
                    # If the struct is empty or all 0 we mark the piece to delete
                    if not tetr.struct.any():
                        tetr_to_remove.append(tetr)
                # Remove the marked pieces to remove from played_tetrominoes and clear the list
                if tetr_to_remove:
                    for tetr in tetr_to_remove:
                        self.played_tetr.remove(tetr)
                    tetr_to_remove.clear()

                # Update the position for the affected tetrominoes by the line clear, i.e. all above the row.
                for tetr in self.played_tetr:
                    rows, _ = np.where(tetr.struct == 1)
                    real_y = tetr._y + min(rows)
                    if real_y <= row:
                        tetr.move(DOWN)
        return affected_rows.size

    @staticmethod
    def _tetr_in_row(tetrominoes, row):
        affected_tetr = []
        for tetr in tetrominoes:
            tetr_struct = np.where(tetr.struct)
            tetr_struct_shifted = tetr_struct[0] + tetr._y, tetr_struct[1] + tetr._x
            if np.any(tetr_struct_shifted[0] == row):
                affected_tetr.append(tetr)
        return affected_tetr

    def _collision(self, tetr):
        """
        Return True if a tetromino collides with other played tetromino or out of board, else return False
        """
        # Check if out of board (left, right and bottom edges)
        if tetr.left < 0 or tetr.right > BOARD_WIDTH-1 or tetr.bot > BOARD_HEIGHT-1:
            return True

        # Current board where to check collision.
        # Note: board size is (21,10), because some tetrominoes can spawn one line above the visible board
        # temp_board = np.vstack((np.zeros((1, BOARD_WIDTH), dtype=int), self.board))

        temp_board = np.zeros((BOARD_HEIGHT + 1, BOARD_WIDTH), dtype=int)
        temp_board[1:] = self.get_board()
        # temp_board = np.zeros((BOARD_HEIGHT+1, BOARD_WIDTH), dtype=int)
        # for tetr_placed in self.played_tetr:
        #     temp_board = TetrisState._place_tetromino(tetr_placed, temp_board)

        # Place ghost_tetr on new position, if there is a collision there will be at least
        # one value greater than 1 (its value will be exactly 2)
        # y position is lowered one cell due to the temp_board being 1 cell taller
        current_struct = np.where(tetr.struct)
        current_struct = current_struct[0] + tetr._y + 1, current_struct[1] + tetr._x
        temp_board[current_struct] += 1
        if np.any(temp_board > 1):
            return True
        return False

    @staticmethod
    def _place_tetromino(tetr: Tetromino, board: np.array):
        new_board = np.copy(board)
        current_struct = np.where(tetr.struct)
        current_struct = current_struct[0] + tetr._y, current_struct[1] + tetr._x
        new_board[current_struct] += 1
        return new_board

    def _move(self, tetr, direction):
        '''
        Moves playing tetromino in specified direction and returns False if there were no collision and True if the
        wanted movement collides and cannot be done
        :param direction: int numpy array of size 2
        :return: bool
        '''
        # Resolve movement
        tetr.move(direction)
        collision = self._collision(tetr)
        # If collision, undo movement
        if collision:
            tetr.move(-direction)
        return collision

    def _drop(self, tetr):
        tetr.status = 'locked'
        collision = self._move(tetr, DOWN)
        score = 0
        while not collision:
            score += 2 * DISTANCE_SCORE
            collision = self._move(tetr, DOWN)
        return score

    def _rotate(self, tetr, rot_direction):
        """
        Rotates current tetromino in specifiied direction and returns False if there were no collision and True if the
        wanted movement collides and cannot be done
        :param rot_direction: 1 for rotation to left, -1 for rotation to the right
        :return: bool
        """
        # Save current tetromino state for wall_kicks
        _, _, starting_rotation_state = tetr.get_state()

        # Basic rotation.
        tetr.rotate(rot_direction)
        if not self._collision(tetr):
            return False

        # Basic rotation has collision, trying wall kicks from rotated piece
        # Select the wall kick array according to the piece
        if tetr.name == BlockID.I:
            wall_kicks = WALL_KICKS_I
        else:
            wall_kicks = WALL_KICKS_JLTSZ

        rotation_direction = (rot_direction + 1) // 2
        for move_x, move_y in wall_kicks[starting_rotation_state][rotation_direction]:
            tetr.move((move_x, -move_y))
            if not self._collision(tetr):
                return False
            else:
                tetr.move((-move_x, move_y))

        # All wall kicks have collision, undo basic rotation
        tetr.rotate(-rot_direction)
        return True

    def _t_spin(self):
        """
        Checks if there was a T-Spin or not
        :return: bool
        """
        if self.current_tetr.name != BlockID.T:
            return False
        if self.actions_done[self.n_actions-1] != Action.ROT_RIGHT.value and self.actions_done[self.n_actions-1] != Action.ROT_LEFT.value:
            return False
        temp_board = np.ones((BOARD_HEIGHT + 2, BOARD_WIDTH + 2), dtype=int)
        temp_board[1:BOARD_HEIGHT+1, 1:BOARD_WIDTH+1] = self.board
        x, y, _ = self.current_tetr.get_state()
        diagonals = temp_board[y+1, x+1] + temp_board[y+3, x+1] + temp_board[y+3, x+3] + temp_board[y+1, x+3]
        if diagonals < 3:
            return False
        return True

    # Additional actions
    def movement_planning(self, x_pos: int, rot_state: int):
        ghost_tetr = self.current_tetr.copy()
        _, _, current_rot_state = self.current_tetr.get_state()
        movements = []
        rotations = rot_state - current_rot_state
        movements.extend([Action.ROT_RIGHT] * rotations)
        ghost_tetr.rotate(-rotations)
        position = x_pos - ghost_tetr.left
        if position > 0:
            movements.extend([Action.RIGHT] * position)
        elif position < 0:
            movements.extend([Action.LEFT] * np.abs(position))
        movements.append(Action.DROP)
        return movements

    # def copy_state(self):
    #     copy_state = TetrisState()
    #     copy_state.board = copy.copy(self.board)
    #     copy_state.played_tetr = copy.copy(self.played_tetr)
    #     copy_state.reserved_tetr = copy.copy(self.reserved_tetr)
    #     copy_state.next_tetr = copy.copy(self.next_tetr)
    #     copy_state.can_reserve = copy.copy(self.can_reserve)
    #
    #     return copy_state

    def save_state(self):
        next_tetr = []
        for tetr in self.next_tetr:
            next_tetr.append(tetr.name)
        if self.reserved_tetr:
            reserved = self.reserved_tetr.name
        else:
            reserved = None

        state_dict = {
            "board": self.board,
            "current": self.current_tetr.name,
            "next": next_tetr,
            "reserved": reserved
        }

        return state_dict

    # Getters
    def get_board(self):
        return self.board

    def get_reserved(self):
        if self.reserved_tetr is None:
            return 0
        return self.reserved_tetr.name.value

    def get_next_tetrominoes(self):
        next_tetr = np.zeros_like(self.next_tetr)
        for i, tetr in enumerate(self.next_tetr):
            next_tetr[i] = tetr.name.value
        return next_tetr

    def get_current_tetromino(self):
        if self.current_tetr is None:
            return 0
        return self.current_tetr.name.value

    def get_total_lines_cleared(self):
        return self.lines

    def get_score(self):
        return self.score

    def get_pieces_placed(self):
        return self.pieces_placed

    def get_actions(self):
        return self.actions_done

    def get_num_actions(self):
        return self.last_piece_num_actions

    @staticmethod
    def get_board_energy(board: np.array):
        energy = 0
        highest_row_per_column = []
        holes_per_column = []
        height, width = board.shape
        for col in range(width):
            column = board[:, col]
            rows = np.where(column == 1)[0]
            if not np.any(rows):
                highest_row = 0
                holes = []
            else:
                highest_row = 20 - min(rows)

                rows_zeros = 20 - np.where(column == 0)[0]
                holes = [x for x in rows_zeros if x < highest_row]
                np_holes = np.array(holes)
                energy += sum((20 - np_holes) ** 2)

            highest_row_per_column.append(highest_row)
            energy += highest_row ** 2
            holes_per_column.append(holes)

        return energy

    def get_future_states(self):
        initial_state = self.save_state()
        ghost_state = TetrisState()
        ghost_state.load(initial_state)
        test_tetr = ghost_state.current_tetr.copy()
        all_movements = []
        all_board_energies = []
        all_positions = []

        for rot in range(ghost_state.current_tetr.max_rotations):
            columns = BOARD_WIDTH - test_tetr.width + 1
            for col in range(columns):
                movements = ghost_state.movement_planning(col, rot)
                for move in movements:
                    over, piece_locked, lines_cleared = ghost_state.update(move.value)
                    if over or piece_locked:
                        continue
                board_energy = TetrisState.get_board_energy(ghost_state.get_board())
                all_movements.append(movements)
                all_board_energies.append(board_energy)
                all_positions.append((col, rot))
                ghost_state.load(initial_state)
            test_tetr.rotate(-1)

        future_states = {
            "movements": all_movements,
            "board_energies": all_board_energies,
            "positions": all_positions
        }

        return future_states

    # Rendering
    def render_frame(self, canvas, cell_size):
        # Playing surfaces
        main_grid = pygame.Surface((cell_size * 10 + 1, cell_size * 20 + 1))
        main_grid.fill((0, 0, 0))
        reserve_grid = pygame.Surface((cell_size * 2 + 1, cell_size * 1 + 1))
        reserve_grid.fill((0, 0, 0))
        next_grid = pygame.Surface((cell_size * 2 + 1, cell_size * 9 + 1))
        next_grid.fill((0, 0, 0))

        # Info surfaces
        info_grid = pygame.Surface((cell_size * 5 + 1, cell_size * 5 + 1))
        info_font = pygame.font.Font(None, 30)
        score = info_font.render(f'Score: {self.score}', True, (255, 255, 255))
        lines_cleared = info_font.render(f'Lines: {self.lines}', True, (255, 255, 255))
        combo = info_font.render(f'Combo: {self.combo}', True, (255, 255, 255))

        timestamp = time.monotonic_ns() - self.starting_time

        def ns_to_min_s_ms(nanoseconds):
            milliseconds = (nanoseconds // 10**6) % 1000
            seconds = (nanoseconds // 10**9) % 60
            minutes = (nanoseconds // (60*10**9))
            return minutes, seconds, milliseconds
        minutes, seconds, milliseconds = ns_to_min_s_ms(timestamp)

        game_time = info_font.render(f'Time: {minutes:02d}:{seconds:02d}:{milliseconds:03d}', True, (255, 255, 255))
        game_ticks = info_font.render(f'Game ticks: {self.game_ticks}', True, (255, 255, 255))

        info_grid.blit(source=score, dest=(0, 0))
        info_grid.blit(source=lines_cleared, dest=(0, 30))
        info_grid.blit(source=combo, dest=(0, 60))
        info_grid.blit(source=game_time, dest=(0, 90))
        info_grid.blit(source=game_ticks, dest=(0, 120))

        def draw_grid_lines(surface, spacing):
            """
            surface: destination surface where draw the lines
            spacing: pixels between lines (square grid)
            """
            height = surface.get_height() // spacing
            width = surface.get_width() // spacing
            for y in range(height + 1):
                pygame.draw.line(
                    surface=surface,
                    color=(255, 255, 255),
                    start_pos=(0, spacing * y),
                    end_pos=(width * spacing, spacing * y),
                    width=1,
                )
            for x in range(width + 1):
                pygame.draw.line(
                    surface=surface,
                    color=(255, 255, 255),
                    start_pos=(spacing * x, 0),
                    end_pos=(spacing * x, height * spacing),
                    width=1,
                )

        def render_tetr(tetromino, surface, cell_size, main_grid=True, pos_offset=(0, 0)):
            """
            Function to draw tetrominos on differents grid of gameplay.
            tetromino: tetromino to draw on surface.
            surface: surface to draw tetromino onto.
            cell_size: size in pixels of single square of a tetromino.
            main_grid: bool to know if drawing tetromino on main grid or reserved/next grid.
            pos_offset: offset in position when drawing on next grid.
            """
            off_set = 4
            board_position = (0, 0)
            if main_grid:
                x, y, _ = tetromino.get_state()
            else:
                x, y = 0, 0
            rows, cols = np.where(tetromino.struct == 1)
            for row, col in zip(rows, cols):
                pygame.draw.rect(
                    surface=surface,
                    color=tetromino.color,
                    rect=pygame.Rect((col + x) * cell_size + off_set/2 + pos_offset[0],
                                     (row + y) * cell_size + off_set/2 + pos_offset[1],
                                     cell_size - off_set + 1,
                                     cell_size - off_set + 1)
                )

        # Render current tetromino
        render_tetr(self.current_tetr, main_grid, cell_size)

        # Render played tetrominoes
        if self.played_tetr:
            for tetr in self.played_tetr:
                render_tetr(tetr, main_grid, cell_size)

        # Render reserved tetromino
        if self.reserved_tetr:
            render_tetr(self.reserved_tetr, reserve_grid, cell_size//2, False)

        # Render next tetrominoes
        for i, tetr in enumerate(self.next_tetr):
            off_set = 3*i * cell_size//2
            render_tetr(tetr, next_grid, cell_size//2, False, (0, off_set))

        # Finally, add some gridlines
        draw_grid_lines(main_grid, cell_size)
        draw_grid_lines(reserve_grid, cell_size//2)
        draw_grid_lines(next_grid, cell_size//2)

        # Blit the different surfaces onto canvas
        grids_offset = 6
        canvas.blit(source=main_grid, dest=((4+grids_offset)*cell_size, 2*cell_size))
        canvas.blit(source=reserve_grid, dest=((1+grids_offset)*cell_size, 2*cell_size))
        canvas.blit(source=next_grid, dest=((15+grids_offset)*cell_size, 2*cell_size))
        canvas.blit(source=info_grid, dest=(1*cell_size, 4*cell_size))

        return canvas


