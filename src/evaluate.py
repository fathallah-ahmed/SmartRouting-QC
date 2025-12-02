"""
Evaluation and Benchmarking Module

This script evaluates the trained DQN agent against Dijkstra baseline on test data.
Produces comprehensive metrics and comparison results.
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple, Dict

from data_loader import load_quebec_graph, preprocess_graph
from env_routing import RoutingEnv
from agent_rl import DQNAgent
from baseline import dijkstra_path, compute_path_metrics


def evaluate_rl_agent(env: RoutingEnv, agent: DQNAgent, start: int, goal: int,
                      max_steps: int = 500) -> Tuple[List[int], Dict[str, float], bool]:
    """
    Evaluate RL agent on a single route.
    
    Args:
        env: Routing environment
        agent: Trained DQN agent
        start: Starting node
        goal: Goal node
        max_steps: Maximum steps
        
    Returns:
        (path, metrics, success)
    """
    state = env.reset(start, goal)
    
    for step in range(max_steps):
        available_actions = env.get_available_actions()
        action = agent.select_action(state, available_actions, training=False)
        
        next_state, reward, done, info = env.step(action)
        state = next_state
        
        if done:
            break
    
    path = env.path
    metrics = env.get_path_metrics()
    success = env.current_node == env.goal_node
    
    return path, metrics, success


def evaluate(args):
    """Main evaluation function."""
    
    print("="*60)
    print("DQN Route Optimization Evaluation")
    print("="*60)
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Load graph
    print("\n1. Loading Quebec City road network...")
    G = load_quebec_graph(cache_path=args.graph_cache)
    G = preprocess_graph(G,
                        distance_weight=args.distance_weight,
                        time_weight=args.time_weight,
                        quality_weight=args.quality_weight)
    
    # Load train/test splits
    print("\n2. Loading train/test splits...")
    splits_path = os.path.join(args.log_dir, 'train_test_splits.json')
    with open(splits_path, 'r') as f:
        splits = json.load(f)
    
    test_pairs = splits['test_pairs']
    print(f"Loaded {len(test_pairs)} test pairs")
    
    # Create environment
    env = RoutingEnv(G, max_steps=args.max_steps)
    
    # Load trained agent
    print("\n3. Loading trained DQN agent...")
    agent = DQNAgent(
        n_nodes=len(G.nodes()),
        state_dim=5,
        hidden_dim=args.hidden_dim,
    )
    agent.load_model(args.model_path)
    
    # Evaluation
    print(f"\n4. Evaluating on {len(test_pairs)} test pairs...")
    results = []
    
    for start, goal in tqdm(test_pairs, desc="Evaluating"):
        # RL agent
        rl_path, rl_metrics, rl_success = evaluate_rl_agent(env, agent, start, goal, args.max_steps)
        
        # Dijkstra baseline
        dijkstra_path_nodes = dijkstra_path(G, start, goal, weight='cost')
        
        if dijkstra_path_nodes:
            dijkstra_metrics = compute_path_metrics(G, dijkstra_path_nodes)
            dijkstra_success = True
        else:
            dijkstra_metrics = {
                'distance_m': float('inf'),
                'distance_km': float('inf'),
                'time_s': float('inf'),
                'time_min': float('inf'),
                'cost': float('inf'),
                'num_nodes': 0,
            }
            dijkstra_success = False
        
        # Compare
        result = {
            'start': start,
            'goal': goal,
            
            # RL metrics
            'rl_success': rl_success,
            'rl_distance_km': rl_metrics['distance_m'] / 1000.0,
            'rl_time_min': rl_metrics['time_s'] / 60.0,
            'rl_cost': rl_metrics['cost'],
            'rl_num_nodes': rl_metrics['num_nodes'],
            
            # Dijkstra metrics
            'dijkstra_success': dijkstra_success,
            'dijkstra_distance_km': dijkstra_metrics['distance_km'],
            'dijkstra_time_min': dijkstra_metrics['time_min'],
            'dijkstra_cost': dijkstra_metrics['cost'],
            'dijkstra_num_nodes': dijkstra_metrics['num_nodes'],
        }
        
        # Compute ratios (RL / Dijkstra)
        if dijkstra_success and rl_success:
            result['distance_ratio'] = rl_metrics['distance_m'] / (dijkstra_metrics['distance_m'] + 1e-6)
            result['time_ratio'] = rl_metrics['time_s'] / (dijkstra_metrics['time_s'] + 1e-6)
            result['cost_ratio'] = rl_metrics['cost'] / (dijkstra_metrics['cost'] + 1e-6)
        else:
            result['distance_ratio'] = float('nan')
            result['time_ratio'] = float('nan')
            result['cost_ratio'] = float('nan')
        
        results.append(result)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save detailed results
    print(f"\n5. Saving results to {args.output}...")
    df.to_csv(args.output, index=False)
    
    # Compute summary statistics
    print("\n6. Computing summary statistics...")
    
    # Filter successful routes for both methods
    both_success = df[(df['rl_success']) & (df['dijkstra_success'])]
    
    summary = {
        'total_test_pairs': len(test_pairs),
        'rl_success_count': df['rl_success'].sum(),
        'rl_success_rate': df['rl_success'].mean(),
        'dijkstra_success_count': df['dijkstra_success'].sum(),
        'dijkstra_success_rate': df['dijkstra_success'].mean(),
        'both_success_count': len(both_success),
    }
    
    if len(both_success) > 0:
        summary.update({
            # Distance comparison
            'avg_distance_ratio': both_success['distance_ratio'].mean(),
            'std_distance_ratio': both_success['distance_ratio'].std(),
            'median_distance_ratio': both_success['distance_ratio'].median(),
            
            # Time comparison
            'avg_time_ratio': both_success['time_ratio'].mean(),
            'std_time_ratio': both_success['time_ratio'].std(),
            'median_time_ratio': both_success['time_ratio'].median(),
            
            # Cost comparison
            'avg_cost_ratio': both_success['cost_ratio'].mean(),
            'std_cost_ratio': both_success['cost_ratio'].std(),
            'median_cost_ratio': both_success['cost_ratio'].median(),
            
            # Absolute metrics
            'rl_avg_distance_km': both_success['rl_distance_km'].mean(),
            'rl_avg_time_min': both_success['rl_time_min'].mean(),
            'rl_avg_cost': both_success['rl_cost'].mean(),
            
            'dijkstra_avg_distance_km': both_success['dijkstra_distance_km'].mean(),
            'dijkstra_avg_time_min': both_success['dijkstra_time_min'].mean(),
            'dijkstra_avg_cost': both_success['dijkstra_cost'].mean(),
        })
    
    # Save summary
    summary_path = args.output.replace('.csv', '_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved summary to {summary_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"\nTest Pairs: {summary['total_test_pairs']}")
    print(f"\nSuccess Rates:")
    print(f"  RL Agent:  {summary['rl_success_rate']:.2%} ({summary['rl_success_count']}/{summary['total_test_pairs']})")
    print(f"  Dijkstra:  {summary['dijkstra_success_rate']:.2%} ({summary['dijkstra_success_count']}/{summary['total_test_pairs']})")
    print(f"  Both:      {summary['both_success_count']}/{summary['total_test_pairs']}")
    
    if len(both_success) > 0:
        print(f"\nPerformance Ratios (RL / Dijkstra):")
        print(f"  Distance:  {summary['avg_distance_ratio']:.3f} ± {summary['std_distance_ratio']:.3f}")
        print(f"  Time:      {summary['avg_time_ratio']:.3f} ± {summary['std_time_ratio']:.3f}")
        print(f"  Cost:      {summary['avg_cost_ratio']:.3f} ± {summary['std_cost_ratio']:.3f}")
        print(f"\n  (Ratio < 1.0 means RL is better, > 1.0 means Dijkstra is better)")
        
        print(f"\nAverage Metrics (Routes where both succeeded):")
        print(f"\n  RL Agent:")
        print(f"    Distance: {summary['rl_avg_distance_km']:.2f} km")
        print(f"    Time:     {summary['rl_avg_time_min']:.2f} min")
        print(f"    Cost:     {summary['rl_avg_cost']:.4f}")
        
        print(f"\n  Dijkstra:")
        print(f"    Distance: {summary['dijkstra_avg_distance_km']:.2f} km")
        print(f"    Time:     {summary['dijkstra_avg_time_min']:.2f} min")
        print(f"    Cost:     {summary['dijkstra_avg_cost']:.4f}")
    
    print("\n" + "="*60)
    
    return df, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate DQN agent against Dijkstra baseline')
    
    # Data parameters
    parser.add_argument('--graph-cache', type=str, default='data/quebec_graph.pkl',
                       help='Path to cached graph file')
    parser.add_argument('--log-dir', type=str, default='results/logs',
                       help='Directory containing train/test splits')
    
    # Cost weights (should match training)
    parser.add_argument('--distance-weight', type=float, default=0.4,
                       help='Weight for distance in cost function')
    parser.add_argument('--time-weight', type=float, default=0.4,
                       help='Weight for time in cost function')
    parser.add_argument('--quality-weight', type=float, default=0.2,
                       help='Weight for road quality in cost function')
    
    # Model parameters
    parser.add_argument('--model-path', type=str, default='models/final_model.pth',
                       help='Path to trained model')
    parser.add_argument('--hidden-dim', type=int, default=256,
                       help='Hidden layer dimension (must match training)')
    parser.add_argument('--max-steps', type=int, default=500,
                       help='Maximum steps per episode')
    
    # Output parameters
    parser.add_argument('--output', type=str, default='results/evaluation_results.csv',
                       help='Path to save evaluation results')
    
    args = parser.parse_args()
    
    evaluate(args)
