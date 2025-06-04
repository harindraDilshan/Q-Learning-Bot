from CellType import *

# Constants
GRID_SIZE = 5
CELL_SIZE = 80
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
FPS = 10

# Colors
COLORS = {
    CellType.EMPTY: (255, 255, 255),    # White
    CellType.AGENT: (0, 100, 255),      # Blue
    CellType.GEM: (0, 255, 0),          # Green
    CellType.MINE: (255, 0, 0),         # Red
    CellType.BARRIER: (100, 100, 100)   # Gray
}