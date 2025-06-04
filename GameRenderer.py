from constants import *
from GridEnvironment import *
import pygame

class GameRenderer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE + 100))
        pygame.display.set_caption("Q-Learning Grid World")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        
    def render(self, environment: GridEnvironment, episode: int, 
               total_reward: float, epsilon: float):
        """Render the current state of the environment"""
        self.screen.fill((0, 0, 0))  # Black background
        
        # Draw grid
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                cell_type = CellType(environment.grid[row, col])
                color = COLORS[cell_type]
                
                rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, 
                                 CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 2)
        
        # Draw statistics
        stats_y = WINDOW_SIZE + 10
        episode_text = self.font.render(f"Episode: {episode}", True, (255, 255, 255))
        reward_text = self.font.render(f"Reward: {total_reward:.2f}", True, (255, 255, 255))
        epsilon_text = self.font.render(f"Epsilon: {epsilon:.3f}", True, (255, 255, 255))
        
        self.screen.blit(episode_text, (10, stats_y))
        self.screen.blit(reward_text, (200, stats_y))
        self.screen.blit(epsilon_text, (400, stats_y))
        
        pygame.display.flip()
        self.clock.tick(FPS)
    
    def close(self):
        """Close the pygame window"""
        pygame.quit()