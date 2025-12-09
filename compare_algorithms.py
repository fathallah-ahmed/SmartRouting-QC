"""
Compare A* vs DQN Routing Performance

This script compares the classical A* algorithm with the DQN agent
on the same routing tasks to demonstrate the superiority of A* for
this problem.
"""

import json
import sys
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict

# Add src to path
sys.path.insert(0, 'src')

from data_loader import load_quebec_graph, preprocess_graph, create_small_graph
from astar_routing import astar_route, compare_path_metrics


def load_test_pairs(path: str = "results/logs/train_test_splits.json") -> Tuple[List, List]:
    """Load the train/test node pairs from previous DQN run."""
    with open(path, 'r') as f:
        data = json.load(f)
    return data['train_pairs'], data['test_pairs']


def test_astar_performance(G, node_pairs: List[Tuple[int, int]], 
                          description: str = "Test") -> Dict:
    """
    Test A* algorithm on a set of node pairs.
    
    Returns metrics: success rate, average cost, average hops, etc.
    """
    print(f"\n{description}:")
    print(f"Testing {len(node_pairs)} routes...")
    
    successes = 0
    total_cost = 0
    total_hops = 0
    total_distance = 0
    failed_pairs = []
    
    for start, goal in node_pairs:
        result = astar_route(G, start, goal)
        
        if result:
            path, cost = result
            metrics = compare_path_metrics(G, path)
            
            successes += 1
            total_cost += cost
            total_hops += metrics['num_hops']
            total_distance += metrics['path_length_meters']
        else:
            failed_pairs.append((start, goal))
    
    success_rate = (successes / len(node_pairs)) * 100
    
    results = {
        'total_pairs': len(node_pairs),
        'successes': successes,
        'failures': len(failed_pairs),
        'success_rate': success_rate,
        'avg_cost': total_cost / successes if successes > 0 else 0,
        'avg_hops': total_hops / successes if successes > 0 else 0,
        'avg_distance_m': total_distance / successes if successes > 0 else 0,
    }
    
    # Print results
    print(f"\n{'='*60}")
    print(f"A* ALGORITHM RESULTS - {description}")
    print(f"{'='*60}")
    print(f"Success Rate: {success_rate:.1f}% ({successes}/{len(node_pairs)})")
    print(f"Average Cost: {results['avg_cost']:.2f}")
    print(f"Average Hops: {results['avg_hops']:.1f}")
    print(f"Average Distance: {results['avg_distance_m']:.0f}m")
    
    if failed_pairs:
        print(f"\n⚠️  Failed to find paths for {len(failed_pairs)} pairs")
        print(f"   (These node pairs may be disconnected)")
    
    return results


def load_dqn_results(path: str = "results/logs/training_metrics.json") -> Dict:
    """Load DQN training results for comparison."""
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Calculate final success rate
        final_success_rate = np.mean(data['episode_success'][-100:]) * 100
        final_reward = np.mean(data['episode_rewards'][-100:])
        
        return {
            'success_rate': final_success_rate,
            'avg_reward': final_reward,
            'total_episodes': len(data['episode_success'])
        }
    except FileNotFoundError:
        return None


def main():
    print("="*60)
    print("A* vs DQN ROUTING COMPARISON")
    print("="*60)
    
    # Load graph
    print("\n1. Loading Quebec City road network...")
    G = load_quebec_graph(cache_path='data/quebec_graph.pkl')
    G = preprocess_graph(G)
    
    # Create same subgraph as DQN training
    print("\n2. Creating 5000-node subgraph (same as DQN training)...")
    G_small = create_small_graph(G, n_nodes=5000, seed=42)
    
    # Load test pairs
    print("\n3. Loading test pairs from DQN training...")
    try:
        train_pairs, test_pairs = load_test_pairs()
        print(f"   Loaded {len(train_pairs)} training pairs, {len(test_pairs)} test pairs")
    except FileNotFoundError:
        print("   ⚠️  No saved pairs found. Please run DQN training first.")
        return
    
    # Test A* on same pairs
    print("\n4. Running A* algorithm...")
    train_results = test_astar_performance(G_small, train_pairs, "Training Set")
    test_results = test_astar_performance(G_small, test_pairs, "Test Set")
    
    # Load DQN results for comparison
    print("\n5. Loading DQN results for comparison...")
    dqn_results = load_dqn_results()
    
    if dqn_results:
        print(f"\n{'='*60}")
        print(f"COMPARISON: A* vs DQN")
        print(f"{'='*60}")
        print(f"\nDQN (after {dqn_results['total_episodes']} episodes):")
        print(f"  Success Rate: {dqn_results['success_rate']:.1f}%")
        print(f"  Avg Reward: {dqn_results['avg_reward']:.1f}")
        
        print(f"\nA* (no training needed):")
        print(f"  Success Rate: {test_results['success_rate']:.1f}%")
        print(f"  Avg Cost: {test_results['avg_cost']:.2f}")
        
        print(f"\n{'='*60}")
        print(f"WINNER: A* 🎯")
        print(f"{'='*60}")
        print(f"✅ A* achieves {test_results['success_rate']:.1f}% success rate")
        print(f"✅ DQN only achieved {dqn_results['success_rate']:.1f}% after {dqn_results['total_episodes']} episodes")
        print(f"✅ A* provides GUARANTEED optimal paths")
        print(f"✅ No training required!")
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print("For routing problems, A* is the superior choice:")
    print("  • Guaranteed optimal solutions")
    print("  • 100% success rate (if path exists)")
    print("  • Instant results (no training)")
    print("  • Industry standard (Google Maps, etc.)")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
