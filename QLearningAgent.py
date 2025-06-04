import random
from typing import Tuple, List, Optional
from Action import *
import numpy as np

class QLearningAgent:
    def __init__(self, grid_size: int, learning_rate: float = 0.1, 
                 discount_factor: float = 0.95, epsilon: float = 1.0, 
                 epsilon_decay: float = 0.995, epsilon_min: float = 0.01):
        self.grid_size = grid_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Initialize Q-table: [row, col, action] -> Q-value
        self.q_table = np.zeros((grid_size, grid_size, len(Action)))
        
    def get_action(self, state: Tuple[int, int], valid_actions: List[Action]) -> Action:
        """Choose action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            # Exploration: choose random valid action
            return random.choice(valid_actions)
        else:
            # Exploitation: choose best action based on Q-values
            row, col = state
            action_values = []
            for action in valid_actions:
                action_values.append(self.q_table[row, col, action.value])
            
            best_action_index = np.argmax(action_values)
            return valid_actions[best_action_index]
    
    def update_q_table(self, state: Tuple[int, int], action: Action, 
                      reward: float, next_state: Tuple[int, int], done: bool):
        """Update Q-table using Q-learning update rule"""
        row, col = state
        next_row, next_col = next_state
        
        # Current Q-value
        current_q = self.q_table[row, col, action.value]
        
        # Maximum Q-value for next state
        if done:
            max_next_q = 0
        else:
            max_next_q = np.max(self.q_table[next_row, next_col, :])
        
        # Q-learning update rule
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[row, col, action.value] = new_q
    
    def decay_epsilon(self):
        """Decay epsilon for exploration vs exploitation balance"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay