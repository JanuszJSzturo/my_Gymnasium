from enum import Enum
import random
import numpy as np


class BlockID(Enum):
    I = 1
    O = 2
    L = 3
    J = 4
    S = 5
    Z = 6
    T = 7


class Tetromino:
    """
    Tetromino representation, basic movements and helper functions
    Struct: numpy array representing a tetromino structure with dimension (n,n). Easier for managing rotations
    """
    def __init__(self):
        self._rotation_state = 0
        self.struct = self.struct
        self.status = 'playing'

    @staticmethod
    def make(name: BlockID = None):
        """
        Returns a tetromino of type "name" or random tetromino if name is not specified o does not exist
        name: name of the type of tetromino desired if None, random type is returned
        """
        if name == BlockID.J:
            return JBlock()
        elif name == BlockID.S:
            return SBlock()
        elif name == BlockID.I:
            return IBlock()
        elif name == BlockID.L:
            return LBlock()
        elif name == BlockID.Z:
            return ZBlock()
        elif name == BlockID.O:
            return OBlock()
        elif name == BlockID.T:
            return TBlock()

        # If type specified returns a random tetromino
        tetromino = Tetromino.make(random.choice(list(BlockID)))
        return tetromino

    def copy(self):
        '''
        Returns a new copy of the current object
        :return: Tetromino
        '''
        new_tetr = Tetromino.make(self.name)
        x, y, rot = self.get_state()
        new_tetr.set_state(x, y, rot)
        new_tetr.struct = self.struct.copy()
        return new_tetr

    @property
    def left(self):
        '''
        Returns x coordinate of the left-most side
        :return:
        '''
        _, cols = np.where(self.struct == 1)
        return self._x + min(cols)

    @property
    def right(self):
        '''
        Returns x coordinate of the righ-most side
        :return: int
        '''
        _, cols = np.where(self.struct == 1)
        return self._x + max(cols)

    @property
    def top(self):
        '''
        Returns y coordinate of the top-most side
        :return: int
        '''
        rows, _ = np.where(self.struct == 1)
        return self._y + min(rows)

    @property
    def bot(self):
        '''
        Returns y coordinate of the bottom-most side
        :return: int
        '''
        rows, _ = np.where(self.struct == 1)
        return self._y + max(rows)


    @property
    def width(self):
        _, cols = np.where(self.struct == 1)
        return len(np.unique(cols))

    @property
    def height(self):
        rows, _ = np.where(self.struct == 1)
        return len(np.unique(rows))


    def rotate(self, rot_direction):
        self.struct = np.rot90(self.struct, k=rot_direction)  # np.rot90 k>0 counter-clockwise, k<0 clockwise
        self._rotation_state = (self._rotation_state - rot_direction) % 4

    def move(self, direction):
        # Resolve movement
        self._x += direction[0]
        self._y += direction[1]

    def get_state(self):
        return self._x, self._y, self._rotation_state

    def update_struct(self, row):
        new_struct = self.struct
        new_struct = np.delete(new_struct, row, 0)

        return new_struct

    def set_state(self, x, y, rot):
        self._x = x
        self._y = y
        rotation_max = self.rotations
        if 0 <= rot < rotation_max:
            self._rotation_state = rot
        else:
            self._rotation_state = 0
            print(f'Rotation state {rot} invalid for {self.name}')



class IBlock(Tetromino):
    struct = np.array(
        ((0, 0, 0, 0),
         (1, 1, 1, 1),
         (0, 0, 0, 0),
         (0, 0, 0, 0))
    )
    color = (0, 168, 221)
    name = BlockID.I
    rotations = 4
    max_rotations = 2
    _x = 3
    _y = -1


class LBlock(Tetromino):
    struct = np.array(
        ((0, 0, 1),
         (1, 1, 1),
         (0, 0, 0))
    )
    color = (237, 96, 0)
    name = BlockID.L
    rotations = 4
    max_rotations = 4
    _x = 3
    _y = -1


class JBlock(Tetromino):
    struct = np.array(
        ((1, 0, 0),
         (1, 1, 1),
         (0, 0, 0))
    )
    color = (41, 18, 245)
    name = BlockID.J
    rotations = 4
    max_rotations = 4
    _x = 3
    _y = -1


class OBlock(Tetromino):
    struct = np.array(
        ((1, 1),
         (1, 1))
    )
    color = (222, 165, 0)
    name = BlockID.O
    rotations = 1
    max_rotations = 1
    _x = 4
    _y = -1


class SBlock(Tetromino):
    struct = np.array(
        ((0, 1, 1),
         (1, 1, 0),
         (0, 0, 0))
    )
    color = (82, 226, 0)
    name = BlockID.S
    rotations = 4
    max_rotations = 2
    _x = 3
    _y = -1


class TBlock(Tetromino):
    struct = np.array(
        ((0, 1, 0),
         (1, 1, 1),
         (0, 0, 0))
    )
    color = (168, 23, 236)
    name = BlockID.T
    rotations = 4
    max_rotations = 4
    _x = 3
    _y = -1


class ZBlock(Tetromino):
    struct = np.array(
        ((1, 1, 0),
         (0, 1, 1),
         (0, 0, 0))
    )
    color = (249, 38, 52)
    name = BlockID.Z
    rotations = 4
    max_rotations = 2
    _x = 3
    _y = -1


# Piece not in use
class DotBlock(Tetromino):
    struct = np.array(
        ((0, 0, 0),
         (0, 1, 0),
         (0, 0, 0))
    )
    color = (87, 87, 87)
    name = 'Dot'
    rotations = 0
    max_rotations = 1
    _x = 3
    _y = -1


if __name__ == '__main__':
    print(f'running tetromino')
