from enum import Enum

class CellType(Enum):
    EMPTY = 0
    AGENT = 1
    GEM = 2
    MINE = 3
    BARRIER = 4