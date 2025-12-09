"""
A* Pathfinding Algorithm for Route Optimization

This implements the A* algorithm for finding optimal routes in road networks.
A* is the industry standard for routing (used in Google Maps, navigation systems, etc.)

WHY A* INSTEAD OF DQN:
- Guaranteed optimal paths (DQN only approximates)
- No training required (instant results)
- 100% success rate (always finds path if one exists)
- Computationally efficient for routing problems
- Interpretable results (can see exact path taken)
"""

import networkx as nx
import heapq
from typing import List, Tuple, Optional, Dict
import numpy as np


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate haversine distance between two GPS coordinates.
    
    This is used as the heuristic function for A*.
    
    Args:
        lat1, lon1: Coordinates of first point
        lat2, lon2: Coordinates of second point
        
    Returns:
        Distance in meters
    """
    R = 6371000  # Earth radius in meters
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


def astar_route(G: nx.MultiDiGraph, 
                start: int, 
                goal: int,
                weight: str = 'cost') -> Optional[Tuple[List[int], float]]:
    """
    Find optimal route using A* algorithm.
    
    Args:
        G: Road network graph
        start: Starting node ID
        goal: Goal node ID
        weight: Edge attribute to use as cost (default: 'cost')
        
    Returns:
        Tuple of (path, total_cost) if path exists, None otherwise
        path: List of node IDs from start to goal
        total_cost: Total cost of the path
    """
    # Check if nodes exist
    if start not in G or goal not in G:
        return None
    
    # Get goal coordinates for heuristic
    goal_pos = (G.nodes[goal]['y'], G.nodes[goal]['x'])
    
    # Priority queue: (f_score, g_score, node, path)
    # f_score = g_score + h_score (total estimated cost)
    # g_score = actual cost from start to current node
    # h_score = heuristic (estimated cost from current to goal)
    open_set = []
    heapq.heappush(open_set, (0, 0, start, [start]))
    
    # Track best g_score for each node
    g_scores: Dict[int, float] = {start: 0}
    
    # Track visited nodes
    closed_set = set()
    
    while open_set:
        f_score, g_score, current, path = heapq.heappop(open_set)
        
        # Found goal!
        if current == goal:
            return path, g_score
        
        # Already processed this node
        if current in closed_set:
            continue
        
        closed_set.add(current)
        
        # Explore neighbors
        for neighbor in G.successors(current):
            if neighbor in closed_set:
                continue
            
            # Get edge cost (handle MultiDiGraph - take minimum cost edge)
            edge_cost = float('inf')
            for key in G[current][neighbor]:
                edge_data = G[current][neighbor][key]
                cost = edge_data.get(weight, edge_data.get('length', 1.0))
                edge_cost = min(edge_cost, cost)
            
            # Calculate new g_score
            tentative_g_score = g_score + edge_cost
            
            # Skip if we've found a better path to this neighbor
            if neighbor in g_scores and tentative_g_score >= g_scores[neighbor]:
                continue
            
            # Calculate heuristic (straight-line distance to goal)
            neighbor_pos = (G.nodes[neighbor]['y'], G.nodes[neighbor]['x'])
            h_score = haversine_distance(neighbor_pos[0], neighbor_pos[1],
                                        goal_pos[0], goal_pos[1])
            
            # Update best path to neighbor
            g_scores[neighbor] = tentative_g_score
            f_score = tentative_g_score + h_score
            
            # Add to open set
            heapq.heappush(open_set, (f_score, tentative_g_score, neighbor, path + [neighbor]))
    
    # No path found
    return None


def compare_path_metrics(G: nx.MultiDiGraph, path: List[int], weight: str = 'cost') -> Dict[str, float]:
    """
    Calculate metrics for a given path.
    
    Args:
        G: Road network graph
        path: List of node IDs
        weight: Edge attribute to use
        
    Returns:
        Dictionary with path metrics
    """
    if len(path) < 2:
        return {'total_cost': 0, 'num_hops': 0, 'path_length_meters': 0}
    
    total_cost = 0
    total_distance = 0
    
    for i in range(len(path) - 1):
        current = path[i]
        next_node = path[i + 1]
        
        # Get minimum cost edge
        min_cost = float('inf')
        min_length = 0
        
        for key in G[current][next_node]:
            edge_data = G[current][next_node][key]
            cost = edge_data.get(weight, edge_data.get('length', 1.0))
            length = edge_data.get('length', 0)
            
            if cost < min_cost:
                min_cost = cost
                min_length = length
        
        total_cost += min_cost
        total_distance += min_length
    
    return {
        'total_cost': total_cost,
        'num_hops': len(path) - 1,
        'path_length_meters': total_distance
    }


if __name__ == "__main__":
    # Test the A* algorithm
    print("Testing A* routing algorithm...")
    
    from data_loader import load_quebec_graph, preprocess_graph
    
    # Load graph
    print("\nLoading Quebec City graph...")
    G = load_quebec_graph()
    G = preprocess_graph(G)
    
    print(f"Graph loaded: {len(G.nodes)} nodes, {len(G.edges)} edges")
    
    # Pick random start and goal
    nodes = list(G.nodes())
    start = nodes[0]
    goal = nodes[100]
    
    print(f"\nFinding route from {start} to {goal}...")
    
    result = astar_route(G, start, goal)
    
    if result:
        path, cost = result
        metrics = compare_path_metrics(G, path)
        
        print(f"\n[SUCCESS] Path found!")
        print(f"   Hops: {metrics['num_hops']}")
        print(f"   Cost: {cost:.2f}")
        print(f"   Distance: {metrics['path_length_meters']:.0f}m")
        print(f"   Path: {' -> '.join(map(str, path[:5]))}... (showing first 5 nodes)")
    else:
        print("\n[FAILED] No path found")
