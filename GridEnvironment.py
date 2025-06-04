from constants import *
from Action import *
import numpy as np
from typing import Tuple, List, Optional

class GridEnvironment:
    def __init__(self):
        self.grid_size = GRID_SIZE
        self.reset()

    def reset(self) -> Tuple[int, int]:
        """Reset the environment to initial state"""
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)

        # Place barriers (walls)
        barriers = [(1, 1), (1, 3), (3, 1), (3, 3)]
        for pos in barriers:
            self.grid[pos] = CellType.BARRIER.value

        # Place mines
        mines = [(2, 1), (1, 4), (4, 2)]
        for pos in mines:
            self.grid[pos] = CellType.MINE.value

        # Place gem
        self.gem_pos = (4, 4)
        self.grid[self.gem_pos] = CellType.GEM.value

        # Place agent
        self.agent_pos = (0, 0)
        self.grid[self.agent_pos] = CellType.AGENT.value

        return self.agent_pos
    
    def get_valid_actions(self, pos: Tuple[int, int]) -> List[Action]:
        """Get valid actions from current position"""
        valid_actions = []
        row, col = pos
        
        # Check each direction
        if row > 0 and self.grid[row-1, col] != CellType.BARRIER.value:
            valid_actions.append(Action.UP)
        if row < self.grid_size-1 and self.grid[row+1, col] != CellType.BARRIER.value:
            valid_actions.append(Action.DOWN)
        if col > 0 and self.grid[row, col-1] != CellType.BARRIER.value:
            valid_actions.append(Action.LEFT)
        if col < self.grid_size-1 and self.grid[row, col+1] != CellType.BARRIER.value:
            valid_actions.append(Action.RIGHT)
            
        return valid_actions
    
    def step(self, action: Action) -> Tuple[Tuple[int, int], float, bool]:
        """Execute action and return new state, reward, and done flag"""
        old_pos = self.agent_pos
        new_pos = self._get_new_position(self.agent_pos, action)
        
        # Check if move is valid
        if not self._is_valid_position(new_pos):
            return self.agent_pos, -0.1, False  # Small penalty for invalid move
        
        # Update agent position
        self.grid[old_pos] = CellType.EMPTY.value
        self.agent_pos = new_pos
        
        # Check what's at the new position
        cell_type = self._get_cell_type(new_pos)
        reward = self._calculate_reward(cell_type)
        done = self._is_terminal(cell_type)
        
        # Update grid
        if not done or cell_type != CellType.MINE:
            self.grid[new_pos] = CellType.AGENT.value
            
        return self.agent_pos, reward, done
    
    def _get_new_position(self, pos: Tuple[int, int], action: Action) -> Tuple[int, int]:
        """Calculate new position based on action"""
        row, col = pos
        if action == Action.UP:
            return (row - 1, col)
        elif action == Action.DOWN:
            return (row + 1, col)
        elif action == Action.LEFT:
            return (row, col - 1)
        elif action == Action.RIGHT:
            return (row, col + 1)
        return pos
    
    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within bounds and not a barrier"""
        row, col = pos
        if row < 0 or row >= self.grid_size or col < 0 or col >= self.grid_size:
            return False
        return self.grid[row, col] != CellType.BARRIER.value
    
    def _get_cell_type(self, pos: Tuple[int, int]) -> CellType:
        """Get the type of cell at position"""
        return CellType(self.grid[pos])
    
    def _calculate_reward(self, cell_type: CellType) -> float:
        """Calculate reward based on cell type"""
        if cell_type == CellType.GEM:
            return 10.0  # Big reward for reaching goal
        elif cell_type == CellType.MINE:
            return -10.0  # Big penalty for hitting mine
        else:
            return -0.01  # Small penalty for each step (encourages efficiency)
    
    def _is_terminal(self, cell_type: CellType) -> bool:
        """Check if current cell is terminal"""
        return cell_type in [CellType.GEM, CellType.MINE]
    