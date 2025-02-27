"""
Created by Teocci.
Author: teocci@yandex.com
Date: 2025-2월-27
"""

import logging
import os
import numpy as np

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='agent.log'
)
logger = logging.getLogger(__name__)

class GameLearningAgent:
    """
    Reinforcement learning agent to play the Monanimal Mayhem game.
    Uses a simple Q-learning approach for demonstration.
    """

    def __init__(self, game_interface, action_space=5, learning_rate=0.01, discount_factor=0.99, exploration_rate=1.0):
        """
        Initialize the reinforcement learning agent.

        Args:
            game_interface: GameInterface instance
            action_space: Number of possible actions
            learning_rate: Alpha parameter for Q-learning
            discount_factor: Gamma parameter for Q-learning
            exploration_rate: Initial epsilon for epsilon-greedy policy
        """
        self.game = game_interface
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = 0.01
        self.exploration_decay = 0.995

        # Create a simple Q-table for learning
        # In a real implementation, you would use a neural network for Q-function approximation
        # For simplicity, we're using a placeholder approach here
        self.q_values = {}  # Will use feature hashing for state representation

        self.total_rewards = []
        self.episode_lengths = []

    def get_state_key(self, state):
        """
        Convert the state representation to a key for the Q-table.
        Uses feature hashing to reduce dimensionality.
        """
        # For simplicity, just use a basic hashing of the state
        # In a real implementation, you would use a more sophisticated state representation
        flattened = state.flatten()
        # Downsample to reduce state space
        downsampled = flattened[::20]
        # Discretize values to reduce state space further
        discretized = (downsampled * 10).astype(int)
        # Convert to tuple for hashing
        return tuple(discretized)

    def choose_action(self, state):
        """
        Choose action using epsilon-greedy policy.

        Args:
            state: Current state representation

        Returns:
            Selected action index
        """
        # Exploration: random action
        if np.random.random() < self.exploration_rate:
            return np.random.randint(self.action_space)

        # Exploitation: best action from Q-table
        state_key = self.get_state_key(state)

        # If state not in Q-table, initialize with zeros
        if state_key not in self.q_values:
            self.q_values[state_key] = np.zeros(self.action_space)

        # Return action with highest Q-value
        return np.argmax(self.q_values[state_key])

    def update_q_value(self, state, action, reward, next_state, done):
        """
        Update Q-value for a state-action pair using Q-learning.

        Args:
            state: Current state representation
            action: Action taken
            reward: Reward received
            next_state: Next state representation
            done: Whether episode is done
        """
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)

        # Initialize Q-values if not in table
        if state_key not in self.q_values:
            self.q_values[state_key] = np.zeros(self.action_space)
        if next_state_key not in self.q_values:
            self.q_values[next_state_key] = np.zeros(self.action_space)

        # Q-learning update
        current_q = self.q_values[state_key][action]

        # Terminal state has no future rewards
        if done:
            max_next_q = 0
        else:
            max_next_q = np.max(self.q_values[next_state_key])

        # Q-learning formula
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)

        # Update Q-value
        self.q_values[state_key][action] = new_q

    def train(self, num_episodes=100, max_steps_per_episode=1000):
        """
        Train the agent using Q-learning.

        Args:
            num_episodes: Number of episodes to train
            max_steps_per_episode: Maximum steps per episode
        """
        logger.info(f"Starting training for {num_episodes} episodes")

        for episode in range(num_episodes):
            # Reset environment
            state = self.game.reset_environment()
            total_reward = 0
            steps = 0

            for step in range(max_steps_per_episode):
                # Choose action
                action = self.choose_action(state)

                # Take action
                next_state, reward, done = self.game.take_action(action)

                # Update Q-value
                self.update_q_value(state, action, reward, next_state, done)

                # Update state and tracking variables
                state = next_state
                total_reward += reward
                steps += 1

                # Break if episode is done
                if done:
                    break

            # Decay exploration rate
            self.exploration_rate = max(self.min_exploration_rate,
                                        self.exploration_rate * self.exploration_decay)

            # Log episode results
            self.total_rewards.append(total_reward)
            self.episode_lengths.append(steps)

            # Log progress
            if episode % 10 == 0:
                avg_reward = np.mean(self.total_rewards[-10:])
                avg_steps = np.mean(self.episode_lengths[-10:])
                logger.info(f"Episode {episode}: Reward={total_reward:.2f}, Steps={steps}, "
                            f"Avg Reward={avg_reward:.2f}, Avg Steps={avg_steps:.2f}, "
                            f"Exploration Rate={self.exploration_rate:.4f}")

                # Save model periodically
                self.save_model(f"model_episode_{episode}.pkl")

    def play(self, num_episodes=5):
        """
        Play the game using the trained policy.

        Args:
            num_episodes: Number of episodes to play
        """
        logger.info(f"Playing {num_episodes} episodes with trained policy")

        for episode in range(num_episodes):
            # Reset environment
            state = self.game.reset_environment()
            total_reward = 0
            steps = 0

            # Set exploration to 0 for evaluation
            original_exploration = self.exploration_rate
            self.exploration_rate = 0

            while True:
                # Choose action using trained policy
                action = self.choose_action(state)

                # Take action
                next_state, reward, done = self.game.take_action(action)

                # Update state and tracking variables
                state = next_state
                total_reward += reward
                steps += 1

                # Break if episode is done
                if done:
                    break

            # Restore exploration rate
            self.exploration_rate = original_exploration

            # Log episode results
            logger.info(f"Episode {episode}: Reward={total_reward:.2f}, Steps={steps}")

    def save_model(self, filename):
        """
        Save the trained Q-values to disk.

        Args:
            filename: File path to save the model
        """
        import pickle

        model_path = os.path.join(DATA_DIR, filename)
        with open(model_path, 'wb') as f:
            pickle.dump(self.q_values, f)

        logger.info(f"Model saved to {model_path}")

    def load_model(self, filename):
        """
        Load Q-values from disk.

        Args:
            filename: File path to load the model from
        """
        import pickle

        model_path = os.path.join(DATA_DIR, filename)
        try:
            with open(model_path, 'rb') as f:
                self.q_values = pickle.load(f)

            logger.info(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False