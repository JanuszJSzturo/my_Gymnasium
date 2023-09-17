from gymnasium.envs.my_tetris.Tetromino import Tetromino, BlockID, DotBlock
from enum import Enum
import numpy as np
import time
import pygame
import random
import copy

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
L1_DELAY_TIME = 500*10**6
L2_DELAY_TIME = 5000*10**6
L3_DELAY_TIME = 20000*10**6

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
    HOLD = 6
    SOFT = 7


# TODO: Comment the code
# TODO: test the game mechanics
# TODO: Define reward function
# TODO: Define observations
# TODO: tetrominoes spawn 2 blocks lower if can
# TODO: configure tetris level

def one_hot_encoding(array, n_classes):
    array = np.array(array)
    if array.size <= 1:
        one_hot_matrix = np.zeros((1, n_classes), dtype=int)
        one_hot_matrix[0][array-1] = 1
        return one_hot_matrix
    one_hot_matrix = np.zeros((array.size, n_classes), dtype=int)
    for i, item in enumerate(array):
        one_hot_matrix[i][item-1] = 1
    return one_hot_matrix


class TetrisState:
    """
    Tetris game representation and interaction
    Board: numpy array of size (20, 10) of int.
        - 1 -> occupied cell
        - 0 -> free cell
    Current, hold -> Tetromino object
    Played, next -> list of Tetromino objects
    """
    def __init__(self):
        """
        Initialize a tetris game state
        Clean the board
        Clean hold tetromino
        Clean played tetrominoes
        Initializes current tetromino with a random tetromino
        Populate future tetrominoes with 6 random tetromino
        """
        self.board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int)
        self.playing_board = np.zeros((BOARD_HEIGHT + 2, BOARD_WIDTH), dtype=int)

        self.played_tetrs = []
        self.hold_tetr = None
        self.next_tetr = []
        self.can_hold = True
        self.time = time.monotonic_ns()


        self.starting_time = time.monotonic_ns()
        self.time_step = DEFAULT_GRAVITY

        self.game_ticks = 0

        self.randomizer_7_bag = list(BlockID)
        random.shuffle(self.randomizer_7_bag)

        self.lines = 0
        self.score = 0
        self.stats = {"single": 0,
                      "double": 0,
                      "triple": 0,
                      "tetris": 0,
                      "t_spin": 0,
                      "t_single": 0,
                      "t_double": 0,
                      "t_triple": 0}

        self.piece_score = 0
        self.pieces_placed = 0
        self.combo = 0
        self.level = 1

        self.actions_done = np.zeros(100)
        self.successful_actions = np.zeros(100)
        self.n_actions = 0
        self.n_successful_actions = 0
        self.last_piece_num_actions = 0

        for n in range(NUM_NEXT_TETR):
            self.next_tetr.append(Tetromino.make(self.randomizer_7_bag.pop()))

        _ = self._spawn_tetromino()

        self.was_t_spin = False

    def reset(self):
        self.__init__()

    def update(self, action):
        self.game_ticks += 1
        game_over = False

        piece_locked_score = 0
        drop_score = 0
        down_score = 0

        dt = time.monotonic_ns()-self.time
        try:
            action = Action(action)
        except ValueError:
            action = None

        if self.current_tetr.status == 'locked':
            game_over = self._spawn_tetromino()
            self.was_t_spin = False

        if action == Action.LEFT:
            success = self._move(self.current_tetr, LEFT)
            if success:
                self.current_tetr.L1_time = time.monotonic_ns()
        elif action == Action.RIGHT:
            success = self._move(self.current_tetr, RIGHT)
            if success:
                self.current_tetr.L1_time = time.monotonic_ns()
        elif action == Action.DOWN:
            success = self._move(self.current_tetr, DOWN)
            if success:
                down_score += DISTANCE_SCORE
        elif action == Action.ROT_LEFT:
            success, wall_kick = self._rotate(self.current_tetr, ROT_LEFT)
            if success:
                self.current_tetr.L2_time = time.monotonic_ns()
                self.current_tetr.L1_time = time.monotonic_ns()
        elif action == Action.ROT_RIGHT:
            success, wall_kick = self._rotate(self.current_tetr, ROT_RIGHT)
            if success:
                self.current_tetr.L2_time = time.monotonic_ns()
                self.current_tetr.L1_time = time.monotonic_ns()
        elif action == Action.DROP:
            drop_score = self._drop(self.current_tetr)
        elif action == Action.HOLD:
            success = self._hold()
            if success:
                self.piece_score = 0
        elif action == Action.SOFT:
            score = self._soft_drop(self.current_tetr)

        # TODO: how should be the priority in gravity over actions?
        if dt >= self.time_step:
            _ = self._move(self.current_tetr, DOWN)
            self.time = time.monotonic_ns()

        # Check for lock delays
        if self._bottom_reached(self.current_tetr):
            dt_L3 = time.monotonic_ns() - self.current_tetr.L3_time
            dt_L2 = time.monotonic_ns() - self.current_tetr.L2_time
            dt_L1 = time.monotonic_ns() - self.current_tetr.L1_time
            if dt_L3 > L3_DELAY_TIME:
                self.current_tetr.lock()

            if self.current_tetr.L2_started:
                if dt_L2 > L2_DELAY_TIME:
                    self.current_tetr.lock()
            else:
                self.current_tetr.L2_started = True
                self.current_tetr.L2_time = time.monotonic_ns()

            if self.current_tetr.L1_started:
                if dt_L1 > L1_DELAY_TIME:
                    self.current_tetr.lock()
            else:
                self.current_tetr.L1_started = True
                self.current_tetr.L1_time = time.monotonic_ns()
        else:
            self.current_tetr.L2_started = False
            self.current_tetr.L1_started = False

        if action is not None:
            self.actions_done[self.n_actions] = int(action.value)
            self.n_actions += 1
        if action is not None:
            if (action != Action.DROP or drop_score != 0) and (action != Action.DOWN or down_score != 0):
                self.successful_actions[self.n_successful_actions] = int(action.value)
                self.n_successful_actions += 1

        # Process lock piece state
        if self.current_tetr.status == 'locked':
            # If current tetromino is locked we check line, spawn a new tetromino and check if it is game over
            self.board = self._place_tetromino(self.current_tetr, self.board)
            self.pieces_placed += 1
            self.played_tetrs.append(self.current_tetr)
            self.was_t_spin = self._t_spin()
            lines_cleared = self._check_line()
            score = 0
            if self.was_t_spin:
                match lines_cleared:
                    case 0:
                        score = 400 / 800 * 10
                        self.stats["t_spin"] += 1
                    case 1:
                        score = 800 / 800 * 10
                        self.stats["t_single"] += 1
                    case 2:
                        score = 1200 / 800 * 10
                        self.stats["t_double"] += 1
                    case 3:
                        score = 1600 / 800 * 10
                        self.stats["t_triple"] += 1
            else:
                match lines_cleared:
                    case 1:
                        score = 100 / 800 * 10
                        self.stats["single"] += 1
                    case 2:
                        score = 300 / 800 * 10
                        self.stats["double"] += 1
                    case 3:
                        score = 400 / 800 * 10
                        self.stats["triple"] += 1
                    case 4:
                        score = 800 / 800 * 10
                        self.stats["tetris"] += 1

            if lines_cleared > 0:
                self.combo += 1
                self.lines += lines_cleared
            else:
                self.combo = 0
            piece_locked_score = (1 + self.combo * 0.5) * (score * self.level)
            self.score += piece_locked_score

            self.can_hold = True

        if self.board[0, :].sum() > 1:
            game_over = True
        if game_over:
            pass
            # print(f'Game Over. Score:{int(self.score)} | Lines: {int(self.lines)}')

        return game_over, self.current_tetr.status, piece_locked_score, self.was_t_spin

    def _update_board(self):
        temp_board = np.zeros((21, 10), dtype=int)
        for tetr in self.played_tetrs:
            temp_struct = np.where(tetr.struct)
            temp_struct = temp_struct[0] + tetr._y, temp_struct[1] + tetr._x
            temp_board[temp_struct] += 1
        self.board = temp_board[:20]

    def _spawn_tetromino(self):
        self.actions_done = np.zeros(100)
        self.successful_actions = np.zeros(100)
        self.last_piece_num_actions = self.n_actions
        self.n_actions = 0
        self.n_successful_actions = 0

        new_current = self.next_tetr.pop(0).name
        if not self.randomizer_7_bag:
            self.randomizer_7_bag = list(BlockID)
            random.shuffle(self.randomizer_7_bag)

        self.next_tetr.append(Tetromino.make(self.randomizer_7_bag.pop()))
        self.current_tetr = Tetromino.make(new_current)

        self.current_tetr.L3_started = True
        self.current_tetr.L3_time = time.monotonic_ns()

        terminated = self._collision(self.current_tetr)

        return terminated

    def _hold(self):
        """
        Holds the playing tetromino if allowed and returns:
         - True if action succeded
         - False if cannot hold the tetromino
        """
        if self.hold_tetr is None:
            self.hold_tetr = Tetromino.make(self.current_tetr.name)
            self._spawn_tetromino()
            self.can_hold = False
            return True
        elif self.can_hold:
            current_tetr_name = self.current_tetr.name
            self.current_tetr = Tetromino.make(self.hold_tetr.name)
            self.hold_tetr = Tetromino.make(current_tetr_name)
            self.can_hold = False
            return True
        return False

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
                affected_tetr = TetrisState._tetr_in_row(self.played_tetrs, row)
                for tetr in affected_tetr:
                    # Delete the row of the struct affected
                    tetr.struct = np.delete(tetr.struct, row-tetr._y, 0)
                    # If the struct is empty or all 0 we mark the piece to delete
                    if not tetr.struct.any():
                        tetr_to_remove.append(tetr)
                # Remove the marked pieces to remove from played_tetrominoes and clear the list
                if tetr_to_remove:
                    for tetr in tetr_to_remove:
                        self.played_tetrs.remove(tetr)
                    tetr_to_remove.clear()
                # Update the position for the affected tetrominoes by the line clear, i.e. all above the row.
                for tetr in self.played_tetrs:
                    rows, _ = np.where(tetr.struct == 1)
                    real_y = tetr._y + min(rows)
                    if real_y <= row:
                        tetr.move(DOWN)
                # self._update_board()
                # TODO: check how to delete completed lines in board instead of iterating on all played tetrominoes to update the board
                self.board = self._update_board_np(affected_rows, self.get_board())
        return affected_rows.size

    def _update_board_np(self, rows, board):
        rows = np.flip(rows)
        temp_board = np.zeros_like(board)
        index = 0
        for row in reversed(range(BOARD_HEIGHT)):
            if index < rows.size:
                if row == rows[index]:
                    index += 1
                    continue
            temp_board[row+index, :] = board[row, :]
        print(temp_board)
        return temp_board

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
        # for tetr_placed in self.played_tetrs:
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
        Moves playing tetromino in specified direction and returns:
            - False if the movement could not be performed
            - True if the movement was successful
        :param direction: int numpy array of size 2
        :return: bool
        '''
        # Resolve movement
        tetr.move(direction)
        # Check for collision
        if self._collision(tetr):
            tetr.move(-direction)
            return False
        else:
            return True

    def _drop(self, tetr):
        score = 0
        success = self._move(tetr, DOWN)
        while success:
            score += 2 * DISTANCE_SCORE
            success = self._move(tetr, DOWN)
        tetr.lock()
        return score

    def _soft_drop(self, tetr):
        score = 0
        success = self._move(tetr, DOWN)
        while success:
            score += 1 * DISTANCE_SCORE
            success = self._move(tetr, DOWN)
        return score

    def _rotate(self, tetr, rot_direction):
        """
        Rotates current tetromino in specifiied direction and returns a bool and wall kick performed if any:
            - False if the rotation could not be performed and None
            - True if the rotation was successful and wall_kick performed if there was no wall_kick it return None
        :param rot_direction: 1 for rotation to left, -1 for rotation to the right
        :return: bool
        """
        # Save current tetromino state for wall_kicks
        _, _, starting_rotation_state = tetr.get_state()

        # Basic rotation.
        tetr.rotate(rot_direction)
        if not self._collision(tetr):
            return True, None

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
                return True, (move_x, -move_y)
            else:
                tetr.move((-move_x, move_y))

        # All wall kicks have collision, undo basic rotation
        tetr.rotate(-rot_direction)
        return False, None

    def _bottom_reached(self, tetr):
        success = self._move(tetr, DOWN)
        if success:
            self._move(tetr, -DOWN)
            return False
        else:
            return True

    def _t_spin(self):
        """
        Checks if there was a T-Spin or not
        :return: bool
        """
        x, y, _ = self.current_tetr.get_state()

        if y <= 1:
            return False
        if self.current_tetr.name != BlockID.T:
            return False
        last_action = self.successful_actions[self.n_successful_actions-1]
        if last_action != Action.ROT_RIGHT.value and last_action != Action.ROT_LEFT.value:
            return False
        temp_board = np.ones((BOARD_HEIGHT + 2, BOARD_WIDTH + 2), dtype=int)
        temp_board[1:BOARD_HEIGHT+1, 1:BOARD_WIDTH+1] = self.board
        diagonals = temp_board[y+1, x+1] + temp_board[y+3, x+1] + temp_board[y+3, x+3] + temp_board[y+1, x+3]
        if diagonals < 3:
            return False
        return True

    def save_state(self, save_played_tetrs=True):
        if save_played_tetrs:
            played_tetrs = self.get_played_tetrominoes()
        else:
            played_tetrs = None
        state_dict = {
            "board": self.get_board(),
            "current": self.get_current_tetromino(),
            "next": self.get_next_tetrominoes(),
            "hold": self.get_hold(),
            "played_tetrs": played_tetrs
        }

        return state_dict

    def load_state(self, game_state: dict):
        self.reset()

        if game_state["played_tetrs"] is None:
            rows, cols = np.where(game_state["board"] == 1)
            for r, c in zip(rows, cols):
                dot = DotBlock()
                x, y, _ = dot.get_state()
                dot.move((c-1-x, r-1-y))
                self.played_tetrs.append(dot)
        else:
            for tetr in game_state["played_tetrs"]:
                self.played_tetrs.append(tetr.copy())

        if game_state["current"]:
            self.current_tetr = Tetromino.make(BlockID(game_state["current"]))

        next_tetr = game_state["next"][:NUM_NEXT_TETR]
        for i, value in enumerate(next_tetr):
            self.next_tetr[i] = Tetromino.make(BlockID(value))

        self._update_board()

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

    # Getters
    def get_board(self):
        return np.array(self.board, dtype=int)

    def get_hold(self):
        if self.hold_tetr is None:
            return 0
        return self.hold_tetr.name.value

    def get_next_tetrominoes(self):
        next_tetr = np.zeros_like(self.next_tetr, dtype=int)
        for i, tetr in enumerate(self.next_tetr):
            next_tetr[i] = tetr.name.value
        return next_tetr

    def get_current_tetromino(self):
        if self.current_tetr is None:
            return 0
        return self.current_tetr.name.value

    def get_played_tetrominoes(self):
        played_tetrs = list()
        if not self.played_tetrs:
            return None

        for tetr in self.played_tetrs:
            played_tetrs.append(tetr.copy())
        return played_tetrs

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
        start = time.time()
        initial_state = self.save_state(save_played_tetrs=False)
        ghost_state = TetrisState()
        ghost_state.load_state(initial_state)

        test_tetr = ghost_state.current_tetr.copy()
        all_movements = []
        all_board_energies = []
        all_positions = []

        for rot in range(ghost_state.current_tetr.max_rotations):
            columns = BOARD_WIDTH - test_tetr.width + 1
            for col in range(columns):
                movements = ghost_state.movement_planning(col, rot)
                for move in movements:
                    game_over, piece_locked, lines_cleared, was_t_spin = ghost_state.update(move.value)
                board_energy = TetrisState.get_board_energy(ghost_state.get_board())
                all_movements.append(movements)
                all_board_energies.append(board_energy)
                all_positions.append((col, rot))
                ghost_state.load_state(initial_state)
            test_tetr.rotate(-1)
        if True:
            ghost_state.load_state(initial_state)
            if ghost_state.current_tetr.name == BlockID.T:
                add_moves = [Action.ROT_LEFT, Action.ROT_RIGHT]
                basic_movements = copy.deepcopy(all_movements)
                for add_move in add_moves:
                    for movements in basic_movements:
                        new_movements = copy.deepcopy(movements)
                        new_movements.pop()
                        new_movements.append(Action.SOFT)
                        new_movements.append(add_move)
                        new_movements.append(Action.DROP)
                        for moves in new_movements:
                            game_, piece_locked, lines_cleared, was_t_spin = ghost_state.update(moves.value)

                        if was_t_spin:
                            board_energy = TetrisState.get_board_energy(ghost_state.get_board())
                            all_movements.append(new_movements)
                            all_board_energies.append(board_energy)
                            x, y, rot_state = ghost_state.played_tetrs[-1].get_state()
                            all_positions.append((x, rot_state))
                        ghost_state.load_state(initial_state)

        future_states = {
            "movements": all_movements,
            "board_energies": all_board_energies,
            "positions": all_positions
        }
        print(f'Total time {time.time() - start}')
        return future_states

    # AI specific methods
    @staticmethod
    def get_state_dqn_conv2d(game_state):
        return TetrisState.get_main_grid_np_dqn(game_state), TetrisState.get_hold_next_np_dqn(game_state)

    @staticmethod
    def get_main_grid_np_dqn(game_state):
        board = game_state["board"]
        board = np.reshape(board, [1, BOARD_HEIGHT, BOARD_WIDTH, 1])
        board = board > 0
        return board

    @staticmethod
    def get_hold_next_np_dqn(game_state):
        current = game_state["current"]
        hold = game_state["hold"]
        next = game_state["next"]

        current = one_hot_encoding(current, n_classes=len(BlockID))
        hold = one_hot_encoding(hold, n_classes=len(BlockID))
        next = one_hot_encoding(next, n_classes=len(BlockID))


        heights_col, holes_col = TetrisState.get_heights_and_holes(game_state["board"])
        holes = np.zeros(10, dtype=int)
        for i, hole in enumerate(holes_col):
            if hole:
                holes[i] = 20 - max(hole)
        heights_col = np.array(heights_col, dtype=int)
        current_hold_next = np.reshape(np.concatenate((current, hold, next)), -1)
        heights_holes = np.reshape(np.concatenate((heights_col, holes)), -1)

        return np.reshape(np.concatenate((heights_holes, current_hold_next)), [1, -1])

    @staticmethod
    def get_heights_and_holes(board: np.array):
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

            highest_row_per_column.append(highest_row)
            holes_per_column.append(holes)
        return highest_row_per_column, holes_per_column

    def get_all_possible_states_conv2d(self):
        game_states, moves, add_scores, dones, is_include_hold, is_new_hold = self.get_all_possible_gamestates(self.save_state())

        main_boards = []
        hold_next_boards = []
        for game_state in game_states:
            in1, in2 = TetrisState.get_state_dqn_conv2d(game_state)
            main_boards.append(in1)
            hold_next_boards.append(in2)

        return [np.concatenate(main_boards), np.concatenate(hold_next_boards)], np.array([add_scores]).reshape(
            [len(add_scores), 1]), dones, is_include_hold, is_new_hold, moves

    def get_all_possible_gamestates(self, state=None):
        all_states = []
        all_movements = []
        all_add_scores = []
        all_dones = []

        ghost_state = TetrisState()
        if state is None:
            initial_state = self.save_state()
        else:
            initial_state = state

        ghost_state.load_state(initial_state)
        test_tetr = ghost_state.current_tetr.copy()

        for rot in range(ghost_state.current_tetr.max_rotations):
            columns = BOARD_WIDTH - test_tetr.width + 1
            for col in range(columns):
                movements = ghost_state.movement_planning(col, rot)
                for move in movements:
                    game_over, piece_locked, add_score, was_t_spin = ghost_state.update(move.value)
                all_states.append(ghost_state.save_state())
                all_movements.append(movements)
                all_add_scores.append(add_score)
                all_dones.append(game_over)

                ghost_state.load_state(initial_state)

            test_tetr.rotate(-1)

        ghost_state.load_state(initial_state)
        if ghost_state.current_tetr.name == BlockID.T:
            add_moves = [Action.ROT_LEFT, Action.ROT_RIGHT]
            basic_movements = copy.deepcopy(all_movements)
            for add_move in add_moves:
                for movements in basic_movements:
                    new_movements = copy.deepcopy(movements)
                    new_movements.pop()
                    new_movements.append(Action.SOFT)
                    new_movements.append(add_move)
                    new_movements.append(Action.DROP)
                    for moves in new_movements:
                        game_over, piece_locked, add_score, was_t_spin = ghost_state.update(moves.value)
                    if was_t_spin:
                        all_states.append(ghost_state.save_state())
                        all_movements.append(movements)
                        all_add_scores.append(add_score)
                        all_dones.append(game_over)

                    ghost_state.load_state(initial_state)

        is_include_hold = False
        is_new_hold = False

        return all_states, all_movements, all_add_scores, all_dones, is_include_hold, is_new_hold

    # Rendering
    def render_frame(self, canvas, cell_size):
        # Playing surfaces
        main_grid = pygame.Surface((cell_size * 10 + 1, cell_size * 20 + 1))
        main_grid.fill((0, 0, 0))
        hold_grid = pygame.Surface((cell_size * 2 + 1, cell_size * 1 + 1))
        hold_grid.fill((0, 0, 0))
        next_grid = pygame.Surface((cell_size * 2 + 1, cell_size * 9 + 1))
        next_grid.fill((0, 0, 0))

        # Info surfaces
        info_grid = pygame.Surface((cell_size * 5 + 1, cell_size * 9 + 1))
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
        stats_1_header = info_font.render(f'S | D | T | Tetris', True, (255, 255, 255))
        stats_2_header = info_font.render(f't-spin | s | d | t', True, (255, 255, 255))
        stats_1 = info_font.render(f'{self.stats["single"]} | {self.stats["double"]} | {self.stats["triple"]} | {self.stats["tetris"]}', True, (255, 255, 255))
        stats_2 = info_font.render(f'{self.stats["t_spin"]} | {self.stats["t_single"]} | {self.stats["t_double"]} | {self.stats["t_triple"]}', True, (255, 255, 255))

        info_grid.blit(source=score, dest=(0, 0))
        info_grid.blit(source=lines_cleared, dest=(0, 30))
        info_grid.blit(source=combo, dest=(0, 60))
        info_grid.blit(source=game_time, dest=(0, 90))
        info_grid.blit(source=game_ticks, dest=(0, 120))

        info_grid.blit(source=stats_1_header, dest=(0, 170))
        info_grid.blit(source=stats_1, dest=(0, 200))
        info_grid.blit(source=stats_2_header, dest=(0, 240))
        info_grid.blit(source=stats_2, dest=(0, 270))

        def draw_grid_lines(surface, spacing):
            """
            surface: destination surface where draw the lines
            spacing: pixels between lines (assuming square grid)
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
            Function to draw tetrominos on differents grids of gameplay.
            tetromino: tetromino to draw on surface.
            surface: surface to draw tetromino on.
            cell_size: size in pixels of single square of a tetromino.
            main_grid: bool to know if drawing tetromino on main grid or hold/next grid.
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
        if self.played_tetrs:
            for tetr in self.played_tetrs:
                render_tetr(tetr, main_grid, cell_size)

        # Render hold tetromino
        if self.hold_tetr:
            render_tetr(self.hold_tetr, hold_grid, cell_size // 2, False)

        # Render next tetrominoes
        for i, tetr in enumerate(self.next_tetr):
            off_set = 3*i * cell_size//2
            render_tetr(tetr, next_grid, cell_size//2, False, (0, off_set))

        # Finally, add some gridlines
        draw_grid_lines(main_grid, cell_size)
        draw_grid_lines(hold_grid, cell_size//2)
        draw_grid_lines(next_grid, cell_size//2)

        # Blit the different surfaces onto canvas
        grids_offset = 6
        canvas.blit(source=main_grid, dest=((4+grids_offset)*cell_size, 2*cell_size))
        canvas.blit(source=hold_grid, dest=((1+grids_offset)*cell_size, 2*cell_size))
        canvas.blit(source=next_grid, dest=((15+grids_offset)*cell_size, 2*cell_size))
        canvas.blit(source=info_grid, dest=(1*cell_size, 4*cell_size))

        return canvas


