from GridEnvironment import *
from QLearningAgent import *
from GameRenderer import *
import time

class QLearningGridWorld:
    def __init__(self):
        self.environment = GridEnvironment()
        self.agent = QLearningAgent(GRID_SIZE)
        self.renderer = GameRenderer()
        self.episode_rewards = []
        
    def train(self, num_episodes: int = 1000, render: bool = True):
        """Train the agent using Q-Learning"""
        print("Starting Q-Learning training...")
        
        for episode in range(num_episodes):
            state = self.environment.reset()
            total_reward = 0
            steps = 0
            max_steps = 100  # Prevent infinite loops
            
            while steps < max_steps:
                # Get valid actions and choose action
                valid_actions = self.environment.get_valid_actions(state)
                if not valid_actions:
                    break
                    
                action = self.agent.get_action(state, valid_actions)
                
                # Take action
                next_state, reward, done = self.environment.step(action)
                
                # Update Q-table
                self.agent.update_q_table(state, action, reward, next_state, done)
                
                # Update state and reward
                state = next_state
                total_reward += reward
                steps += 1
                
                # Render if requested
                if render and episode % 10 == 0:  # Render every 10th episode
                    self.renderer.render(self.environment, episode, 
                                       total_reward, self.agent.epsilon)
                    
                    # Handle pygame events
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self.renderer.close()
                            return
                
                if done:
                    break
            
            # Decay epsilon
            self.agent.decay_epsilon()
            self.episode_rewards.append(total_reward)
            
            # Print progress
            if episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode}, Average Reward: {avg_reward:.2f}, "
                      f"Epsilon: {self.agent.epsilon:.3f}")
        
        print("Training completed!")
        
    def test_agent(self, num_episodes: int = 10):
        """Test the trained agent"""
        print("Testing trained agent...")
        self.agent.epsilon = 0  # No exploration during testing
        
        success_count = 0
        
        for episode in range(num_episodes):
            state = self.environment.reset()
            steps = 0
            max_steps = 50
            
            while steps < max_steps:
                valid_actions = self.environment.get_valid_actions(state)
                if not valid_actions:
                    break
                    
                action = self.agent.get_action(state, valid_actions)
                state, reward, done = self.environment.step(action)
                
                self.renderer.render(self.environment, episode, 0, 0)
                time.sleep(0.2)  # Slower for visualization
                
                steps += 1
                
                if done:
                    if reward > 0:  # Reached gem
                        success_count += 1
                        print(f"Episode {episode + 1}: SUCCESS in {steps} steps")
                    else:
                        print(f"Episode {episode + 1}: FAILED (hit mine)")
                    break
                    
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.renderer.close()
                        return
        
        print(f"Success rate: {success_count}/{num_episodes} ({success_count/num_episodes*100:.1f}%)")
        
    def close(self):
        """Clean up resources"""
        self.renderer.close()

# Usage example
if __name__ == "__main__":
    # Create and train the Q-Learning agent
    game = QLearningGridWorld()
    
    try:
        # Train the agent
        game.train(num_episodes=500, render=True)
        
        # Test the trained agent
        input("Press Enter to test the trained agent...")
        game.test_agent(num_episodes=5)
        
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        game.close()