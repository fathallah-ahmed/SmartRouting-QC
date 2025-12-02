"""
Baseline Implementation using Dijkstra's Algorithm

This module provides baseline routing using NetworkX's shortest path algorithms.
Used for comparison against the RL agent.
"""

import networkx as nx
from typing import List, Tuple, Dict, Any, Optional


def dijkstra_path(graph: nx.MultiDiGraph, 
                  start: int, 
                  goal: int,
                  weight: str = 'cost') -> Optional[List[int]]:
    """
    Find shortest path using Dijkstra's algorithm.
    
    Args:
        graph: Road network graph
        start: Starting node ID
        goal: Goal node ID
        weight: Edge attribute to use as weight ('cost', 'distance', 'time')
        
    Returns:
        List of node IDs forming the path, or None if no path exists
    """
    try:
        path = nx.shortest_path(graph, start, goal, weight=weight)
        return path
    except nx.NetworkXNoPath:
        return None


def compute_path_metrics(graph: nx.MultiDiGraph, path: List[int]) -> Dict[str, float]:
    """
    Calculate metrics for a given path.
    
    Args:
        graph: Road network graph
        path: List of node IDs forming the path
        
    Returns:
        Dictionary with distance, time, and cost metrics
    """
    if not path or len(path) < 2:
        return {
            'distance_m': 0.0,
            'time_s': 0.0,
            'time_min': 0.0,
            'cost': 0.0,
            'num_nodes': len(path) if path else 0,
        }
    
    total_distance = 0.0
    total_time = 0.0
    total_cost = 0.0
    
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        
        if graph.has_edge(u, v):
            # Get edge with minimum cost (for MultiDiGraph)
            min_cost = float('inf')
            best_edge = None
            
            for key in graph[u][v]:
                cost = graph[u][v][key].get('cost', 1.0)
                if cost < min_cost:
                    min_cost = cost
                    best_edge = graph[u][v][key]
            
            if best_edge:
                total_distance += best_edge.get('distance', 0)
                total_time += best_edge.get('time', 0)
                total_cost += best_edge.get('cost', 0)
    
    return {
        'distance_m': total_distance,
        'distance_km': total_distance / 1000.0,
        'time_s': total_time,
        'time_min': total_time / 60.0,
        'cost': total_cost,
        'num_nodes': len(path),
    }


def find_multiple_paths(graph: nx.MultiDiGraph,
                       start: int,
                       goal: int,
                       k: int = 3) -> List[Tuple[List[int], Dict[str, float]]]:
    """
    Find k shortest paths using different criteria.
    
    Args:
        graph: Road network graph
        start: Starting node ID
        goal: Goal node ID
        k: Number of alternative paths to find
        
    Returns:
        List of (path, metrics) tuples
    """
    paths = []
    
    # Find paths optimizing different criteria
    criteria = ['cost', 'distance', 'time']
    
    for criterion in criteria[:k]:
        path = dijkstra_path(graph, start, goal, weight=criterion)
        if path:
            metrics = compute_path_metrics(graph, path)
            metrics['criterion'] = criterion
            paths.append((path, metrics))
    
    return paths


if __name__ == "__main__":
    # Test the baseline
    print("Testing Dijkstra baseline...")
    
    from data_loader import load_quebec_graph, preprocess_graph
    
    # Load and preprocess graph
    G = load_quebec_graph()
    G = preprocess_graph(G)
    
    # Get two random connected nodes
    nodes = list(G.nodes())
    start, goal = nodes[0], nodes[100]
    
    print(f"\nFinding path from {start} to {goal}...")
    
    # Find shortest path
    path = dijkstra_path(G, start, goal)
    
    if path:
        print(f"Path found with {len(path)} nodes")
        
        # Compute metrics
        metrics = compute_path_metrics(G, path)
        print(f"\nPath metrics:")
        print(f"  Distance: {metrics['distance_km']:.2f} km")
        print(f"  Time: {metrics['time_min']:.2f} minutes")
        print(f"  Cost: {metrics['cost']:.4f}")
        print(f"  Nodes: {metrics['num_nodes']}")
        
        # Find alternative paths
        print(f"\nFinding alternative paths...")
        alt_paths = find_multiple_paths(G, start, goal, k=3)
        
        for i, (alt_path, alt_metrics) in enumerate(alt_paths):
            print(f"\nPath {i+1} (optimized for {alt_metrics['criterion']}):")
            print(f"  Distance: {alt_metrics['distance_km']:.2f} km")
            print(f"  Time: {alt_metrics['time_min']:.2f} minutes")
            print(f"  Cost: {alt_metrics['cost']:.4f}")
    else:
        print("No path found!")
    
    print("\nBaseline test complete!")
