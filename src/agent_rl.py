"""
Deep Q-Learning Agent for Route Optimization

This module implements a DQN agent for learning optimal routing policies.

WHY DQN INSTEAD OF TABULAR Q-LEARNING:
- Quebec City's road network has thousands of nodes, making tabular Q-Learning 
  impractical due to state space explosion (O(N²) for all start-goal pairs)
- DQN uses neural network function approximation to generalize across similar 
  road configurations and learn patterns in routing
- Can handle continuous state features (distance to goal, road attributes)
- More scalable for real-world applications with large graphs
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
from typing import List, Tuple, Dict, Any
import os


# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class QNetwork(nn.Module):
    """
    Neural network for Q-value approximation.
    
    Input: State features (current_idx, goal_idx, distance_to_goal, steps, visited_count)
    Output: Q-values for each possible action (node)
    """
    
    def __init__(self, state_dim: int, n_nodes: int, hidden_dim: int = 256):
        """
        Initialize Q-network.
        
        Args:
            state_dim: Dimension of state features
            n_nodes: Total number of nodes (action space size)
            hidden_dim: Hidden layer dimension
        """
        super(QNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_nodes)
        )
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.network(x)


class ReplayBuffer:
    """Experience replay buffer for stable training."""
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """Add an experience to the buffer."""
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample a batch of experiences."""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        """Return current buffer size."""
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Learning agent for route optimization.
    
    Uses:
    - Neural network for Q-value approximation
    - Experience replay for stable training
    - Target network for stable Q-value targets
    - ε-greedy exploration strategy
    """
    
    def __init__(self,
                 n_nodes: int,
                 state_dim: int = 5,
                 hidden_dim: int = 256,
                 learning_rate: float = 0.001,
                 gamma: float = 0.95,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 buffer_capacity: int = 10000,
                 batch_size: int = 64,
                 target_update_freq: int = 10,
                 device: str = None):
        """
        Initialize DQN agent.
        
        Args:
            n_nodes: Total number of nodes in the graph
            state_dim: Dimension of state features
            hidden_dim: Hidden layer dimension
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Exploration decay rate
            buffer_capacity: Replay buffer capacity
            batch_size: Training batch size
            target_update_freq: Frequency of target network updates (episodes)
            device: Device for training ('cuda' or 'cpu')
        """
        self.n_nodes = n_nodes
        self.state_dim = state_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_counter = 0
        
        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Q-networks
        self.q_network = QNetwork(state_dim, n_nodes, hidden_dim).to(self.device)
        self.target_network = QNetwork(state_dim, n_nodes, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # Training metrics
        self.training_losses = []
        self.epsilon_history = []
    
    def state_to_tensor(self, state: Dict[str, Any]) -> torch.Tensor:
        """
        Convert state dictionary to tensor.
        
        Args:
            state: State dictionary from environment
            
        Returns:
            State tensor
        """
        features = np.array([
            state['current_idx'] / self.n_nodes,  # Normalize
            state['goal_idx'] / self.n_nodes,
            state['distance_to_goal'],
            state['steps'] / 500.0,  # Normalize by max steps
            state['visited_count'] / 100.0,  # Normalize
        ], dtype=np.float32)
        
        return torch.FloatTensor(features).to(self.device)
    
    def select_action(self, state: Dict[str, Any], available_actions: List[int], 
                     training: bool = True) -> int:
        """
        Select action using ε-greedy policy.
        
        Args:
            state: Current state
            available_actions: List of valid actions
            training: If True, use exploration; if False, use greedy policy
            
        Returns:
            Selected action (node ID)
        """
        # ε-greedy exploration
        if training and random.random() < self.epsilon:
            return random.choice(available_actions)
        
        # Greedy action selection
        with torch.no_grad():
            state_tensor = self.state_to_tensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor).squeeze(0)
            
            # Mask invalid actions
            mask = torch.full((self.n_nodes,), float('-inf')).to(self.device)
            for action in available_actions:
                # Convert node ID to index
                action_idx = action if isinstance(action, int) else action
                if action_idx < self.n_nodes:
                    mask[action_idx] = 0
            
            masked_q_values = q_values + mask
            
            # Select action with highest Q-value among available actions
            best_action_idx = masked_q_values.argmax().item()
            
            # Find the corresponding node ID
            if best_action_idx in available_actions:
                return best_action_idx
            else:
                # Fallback: return action with highest Q-value from available actions
                best_q = float('-inf')
                best_action = available_actions[0]
                for action in available_actions:
                    if action < self.n_nodes and q_values[action] > best_q:
                        best_q = q_values[action]
                        best_action = action
                return best_action
    
    def store_experience(self, state: Dict[str, Any], action: int, reward: float,
                        next_state: Dict[str, Any], done: bool):
        """Store experience in replay buffer. Action should be node index."""
        state_tensor = self.state_to_tensor(state).cpu().numpy()
        next_state_tensor = self.state_to_tensor(next_state).cpu().numpy()
        self.replay_buffer.push(state_tensor, action, reward, next_state_tensor, done)
    
    def train_step(self) -> float:
        """
        Perform one training step using experience replay.
        
        Returns:
            Training loss
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample batch
        experiences = self.replay_buffer.sample(self.batch_size)
        
        # Prepare batch tensors
        states = torch.FloatTensor(np.array([e.state for e in experiences])).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences])).to(self.device)
        dones = torch.FloatTensor([e.done for e in experiences]).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with current Q-network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.epsilon_history.append(self.epsilon)
    
    def save_model(self, path: str):
        """
        Save model to file.
        
        Args:
            path: Path to save model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'n_nodes': self.n_nodes,
            'state_dim': self.state_dim,
            'training_losses': self.training_losses,
            'epsilon_history': self.epsilon_history,
        }, path)
        
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """
        Load model from file.
        
        Args:
            path: Path to load model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_losses = checkpoint.get('training_losses', [])
        self.epsilon_history = checkpoint.get('epsilon_history', [])
        
        print(f"Model loaded from {path}")
        print(f"Current epsilon: {self.epsilon:.4f}")


if __name__ == "__main__":
    # Test the agent
    print("Testing DQN agent...")
    
    # Create a dummy agent
    agent = DQNAgent(n_nodes=1000, state_dim=5)
    
    # Test state conversion
    dummy_state = {
        'current_idx': 0,
        'goal_idx': 100,
        'distance_to_goal': 0.05,
        'steps': 10,
        'visited_count': 5,
    }
    
    state_tensor = agent.state_to_tensor(dummy_state)
    print(f"\nState tensor shape: {state_tensor.shape}")
    print(f"State tensor: {state_tensor}")
    
    # Test action selection
    available_actions = [1, 2, 3, 4, 5]
    action = agent.select_action(dummy_state, available_actions)
    print(f"\nSelected action: {action}")
    
    # Test experience storage and training
    next_state = dummy_state.copy()
    next_state['steps'] = 11
    
    agent.store_experience(dummy_state, action, -0.5, next_state, False)
    
    # Add more experiences
    for i in range(100):
        agent.store_experience(dummy_state, action, -0.5, next_state, False)
    
    # Train
    loss = agent.train_step()
    print(f"\nTraining loss: {loss:.4f}")
    
    print("\nAgent test complete!")
