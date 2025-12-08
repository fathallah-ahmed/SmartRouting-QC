"""
Training Script for DQN Route Optimization Agent

This script trains a DQN agent on Quebec City's road network and saves the trained model.
Includes train/test split and comprehensive metrics logging.
"""

import argparse
import os
import json
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

from data_loader import load_quebec_graph, preprocess_graph, get_random_node_pairs, split_train_test
from env_routing import RoutingEnv
from agent_rl import DQNAgent


def train_episode(env: RoutingEnv, agent: DQNAgent, start: int, goal: int, 
                 max_steps: int = 500) -> Dict[str, float]:
    """
    Train the agent on a single episode.
    
    Args:
        env: Routing environment
        agent: DQN agent
        start: Starting node
        goal: Goal node
        max_steps: Maximum steps per episode
        
    Returns:
        Episode metrics
    """
    state = env.reset(start, goal)
    total_reward = 0.0
    steps = 0
    losses = []
    
    for step in range(max_steps):
        # Select action
        available_actions = env.get_available_actions()
        action = agent.select_action(state, available_actions, training=True)
        
        # Take step
        next_state, reward, done, info = env.step(action)
        
        # Get valid actions for next state
        next_valid_actions = env.get_available_actions()
        
        # Store experience
        agent.store_experience(state, action, reward, next_state, done, next_valid_actions)
        
        # Train
        loss = agent.train_step()
        if loss > 0:
            losses.append(loss)
        
        total_reward += reward
        steps += 1
        state = next_state
        
        if done:
            break
    
    # Get path metrics
    path_metrics = env.get_path_metrics()
    
    return {
        'total_reward': total_reward,
        'steps': steps,
        'success': env.current_node == env.goal_node,
        'distance_km': path_metrics['distance_m'] / 1000.0,
        'time_min': path_metrics['time_s'] / 60.0,
        'cost': path_metrics['cost'],
        'avg_loss': np.mean(losses) if losses else 0.0,
    }


def train(args):
    """Main training function."""
    
    print("="*60)
    print("DQN Route Optimization Training")
    print("="*60)
    
    # Create output directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Load and preprocess graph
    print("\n1. Loading Quebec City road network...")
    G = load_quebec_graph(cache_path=args.graph_cache)
    G = preprocess_graph(G, 
                        distance_weight=args.distance_weight,
                        time_weight=args.time_weight,
                        quality_weight=args.quality_weight)
    
    # Generate train/test pairs
    print("\n2. Generating train/test node pairs...")
    all_pairs = get_random_node_pairs(G, 
                                     n_pairs=args.n_pairs,
                                     min_distance=args.min_distance,
                                     max_distance=args.max_distance,
                                     seed=args.seed)
    
    train_pairs, test_pairs = split_train_test(all_pairs, 
                                               test_ratio=args.test_ratio,
                                               seed=args.seed)
    
    # Save train/test splits (convert to regular ints for JSON serialization)
    splits_path = os.path.join(args.log_dir, 'train_test_splits.json')
    with open(splits_path, 'w') as f:
        json.dump({
            'train_pairs': [[int(s), int(g)] for s, g in train_pairs],
            'test_pairs': [[int(s), int(g)] for s, g in test_pairs],
        }, f)
    print(f"Saved train/test splits to {splits_path}")
    
    # Create environment and agent
    print("\n3. Initializing environment and agent...")
    env = RoutingEnv(G, max_steps=args.max_steps)
    
    agent = DQNAgent(
        n_nodes=len(G.nodes()),
        state_dim=9, # Added 4 coordinate features
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        target_update_freq=args.target_update_freq,
        device=args.device,
    )
    
    # Training metrics
    episode_rewards = []
    episode_steps = []
    episode_success = []
    episode_losses = []
    
    # Training loop
    print(f"\n4. Training for {args.episodes} episodes...")
    print(f"   Train pairs: {len(train_pairs)}")
    print(f"   Test pairs: {len(test_pairs)}")
    print()
    
    pbar = tqdm(range(args.episodes), desc="Training")
    
    for episode in pbar:
        # Sample random train pair
        start, goal = train_pairs[episode % len(train_pairs)]
        
        # Train episode
        metrics = train_episode(env, agent, start, goal, max_steps=args.max_steps)
        
        # Record metrics
        episode_rewards.append(metrics['total_reward'])
        episode_steps.append(metrics['steps'])
        episode_success.append(1.0 if metrics['success'] else 0.0)
        episode_losses.append(metrics['avg_loss'])
        
        # Update target network
        if (episode + 1) % args.target_update_freq == 0:
            agent.update_target_network()
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Update progress bar
        recent_success_rate = np.mean(episode_success[-100:]) if len(episode_success) >= 100 else np.mean(episode_success)
        recent_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        
        pbar.set_postfix({
            'reward': f'{recent_reward:.1f}',
            'success': f'{recent_success_rate:.2%}',
            'epsilon': f'{agent.epsilon:.3f}',
            'loss': f'{metrics["avg_loss"]:.4f}',
        })
        
        # Save checkpoint
        if (episode + 1) % args.checkpoint_freq == 0:
            checkpoint_path = os.path.join(args.model_dir, f'checkpoint_ep{episode+1}.pth')
            agent.save_model(checkpoint_path)
    
    # Save final model
    final_model_path = os.path.join(args.model_dir, 'final_model.pth')
    agent.save_model(final_model_path)
    
    # Save training metrics
    print("\n5. Saving training metrics...")
    metrics_path = os.path.join(args.log_dir, 'training_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({
            'episode_rewards': episode_rewards,
            'episode_steps': episode_steps,
            'episode_success': episode_success,
            'episode_losses': episode_losses,
            'epsilon_history': agent.epsilon_history,
        }, f)
    print(f"Saved metrics to {metrics_path}")
    
    # Plot training curves
    print("\n6. Plotting training curves...")
    plot_training_curves(episode_rewards, episode_success, episode_losses, 
                        agent.epsilon_history, args.log_dir)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Final model saved to: {final_model_path}")
    print(f"Training metrics saved to: {metrics_path}")
    print(f"Final success rate: {np.mean(episode_success[-100:]):.2%}")
    print(f"Final average reward: {np.mean(episode_rewards[-100:]):.2f}")


def plot_training_curves(rewards: List[float], success: List[float], 
                         losses: List[float], epsilon: List[float], 
                         output_dir: str):
    """Plot and save training curves."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Smooth curves using moving average
    window = 50
    
    def moving_average(data, window):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    # Rewards
    axes[0, 0].plot(rewards, alpha=0.3, label='Raw')
    axes[0, 0].plot(moving_average(rewards, window), label=f'MA({window})')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Success rate
    axes[0, 1].plot(success, alpha=0.3, label='Raw')
    axes[0, 1].plot(moving_average(success, window), label=f'MA({window})')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Success (1=reached goal)')
    axes[0, 1].set_title('Success Rate')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Loss
    axes[1, 0].plot(losses, alpha=0.3, label='Raw')
    axes[1, 0].plot(moving_average(losses, window), label=f'MA({window})')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Training Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Epsilon
    axes[1, 1].plot(epsilon)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Epsilon')
    axes[1, 1].set_title('Exploration Rate (ε)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved training curves to {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DQN agent for route optimization')
    
    # Data parameters
    parser.add_argument('--graph-cache', type=str, default='data/quebec_graph.pkl',
                       help='Path to cached graph file')
    parser.add_argument('--n-pairs', type=int, default=200,
                       help='Number of node pairs to generate')
    parser.add_argument('--min-distance', type=float, default=1000,
                       help='Minimum distance between nodes (meters)')
    parser.add_argument('--max-distance', type=float, default=10000,
                       help='Maximum distance between nodes (meters)')
    parser.add_argument('--test-ratio', type=float, default=0.2,
                       help='Ratio of pairs for testing')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Cost weights
    parser.add_argument('--distance-weight', type=float, default=0.4,
                       help='Weight for distance in cost function')
    parser.add_argument('--time-weight', type=float, default=0.4,
                       help='Weight for time in cost function')
    parser.add_argument('--quality-weight', type=float, default=0.2,
                       help='Weight for road quality in cost function')
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--max-steps', type=int, default=500,
                       help='Maximum steps per episode')
    parser.add_argument('--hidden-dim', type=int, default=256,
                       help='Hidden layer dimension')
    parser.add_argument('--learning-rate', type=float, default=0.0005,
                       help='Learning rate (default: 0.0005)')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor (default: 0.99)')
    parser.add_argument('--epsilon-start', type=float, default=1.0,
                       help='Initial exploration rate')
    parser.add_argument('--epsilon-end', type=float, default=0.01,
                       help='Final exploration rate')
    parser.add_argument('--epsilon-decay', type=float, default=0.9995,
                       help='Exploration decay rate (default: 0.9995 for slow decay)')
    parser.add_argument('--buffer-capacity', type=int, default=50000,
                       help='Replay buffer capacity (default: 50000)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Training batch size')
    parser.add_argument('--target-update-freq', type=int, default=10,
                       help='Target network update frequency (episodes) (default: 10)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    
    # Output parameters
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Directory to save models')
    parser.add_argument('--log-dir', type=str, default='results/logs',
                       help='Directory to save logs')
    parser.add_argument('--checkpoint-freq', type=int, default=500,
                       help='Checkpoint save frequency (episodes)')
    
    args = parser.parse_args()
    
    train(args)
