"""
Data Loader Module for Quebec City Road Network

This module handles downloading and preprocessing OpenStreetMap data for Quebec City.
It provides functions to load the road network graph and enrich it with routing costs.
"""

import os
import pickle
import osmnx as ox
import networkx as nx
import numpy as np
from typing import Tuple, List, Dict


# Default speed limits (km/h) for different road types when maxspeed is not available
DEFAULT_SPEEDS = {
    'motorway': 100,
    'motorway_link': 80,
    'trunk': 90,
    'trunk_link': 70,
    'primary': 70,
    'primary_link': 60,
    'secondary': 60,
    'secondary_link': 50,
    'tertiary': 50,
    'tertiary_link': 40,
    'residential': 40,
    'living_street': 20,
    'unclassified': 40,
    'service': 30,
}

# Road quality weights (higher is better quality)
ROAD_QUALITY = {
    'motorway': 1.0,
    'motorway_link': 0.95,
    'trunk': 0.9,
    'trunk_link': 0.85,
    'primary': 0.8,
    'primary_link': 0.75,
    'secondary': 0.7,
    'secondary_link': 0.65,
    'tertiary': 0.6,
    'tertiary_link': 0.55,
    'residential': 0.5,
    'living_street': 0.4,
    'unclassified': 0.5,
    'service': 0.4,
}


def load_quebec_graph(cache_path: str = "data/quebec_graph.pkl", force_download: bool = False) -> nx.MultiDiGraph:
    """
    Load or download the Quebec City road network from OpenStreetMap.
    
    Args:
        cache_path: Path to cache the downloaded graph
        force_download: If True, re-download even if cache exists
        
    Returns:
        NetworkX MultiDiGraph of Quebec City road network
    """
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    # Try to load from cache
    if not force_download and os.path.exists(cache_path):
        print(f"Loading cached graph from {cache_path}...")
        with open(cache_path, 'rb') as f:
            G = pickle.load(f)
        print(f"Loaded graph with {len(G.nodes)} nodes and {len(G.edges)} edges")
        return G
    
    # Download from OpenStreetMap
    print("Downloading Quebec City road network from OpenStreetMap...")
    print("This may take a few minutes...")
    
    # Download the graph for Quebec City
    # Using drive network to get roads suitable for vehicles
    G = ox.graph_from_place("Quebec City, Canada", network_type='drive')
    
    print(f"Downloaded graph with {len(G.nodes)} nodes and {len(G.edges)} edges")
    
    # Save to cache
    print(f"Saving graph to {cache_path}...")
    with open(cache_path, 'wb') as f:
        pickle.dump(G, f)
    
    return G


def preprocess_graph(G: nx.MultiDiGraph, 
                     distance_weight: float = 0.4,
                     time_weight: float = 0.4,
                     quality_weight: float = 0.2) -> nx.MultiDiGraph:
    """
    Preprocess the graph by adding computed cost attributes to edges.
    
    Cost combines:
    - Distance (length in meters)
    - Time (estimated from distance and speed)
    - Road quality (based on highway type)
    
    Args:
        G: Input graph
        distance_weight: Weight for distance component (0-1)
        time_weight: Weight for time component (0-1)
        quality_weight: Weight for quality component (0-1)
        
    Returns:
        Preprocessed graph with cost attributes
    """
    print("Preprocessing graph...")
    
    # Normalize weights
    total_weight = distance_weight + time_weight + quality_weight
    distance_weight /= total_weight
    time_weight /= total_weight
    quality_weight /= total_weight
    
    # Store normalization factors for later use
    max_distance = 0
    max_time = 0
    
    # First pass: compute raw values and find max for normalization
    for u, v, key, data in G.edges(keys=True, data=True):
        # Get length (distance in meters)
        length = data.get('length', 100)  # Default 100m if missing
        
        # Get or estimate speed
        highway_type = data.get('highway', 'unclassified')
        if isinstance(highway_type, list):
            highway_type = highway_type[0]
        
        # Try to get maxspeed from OSM data
        maxspeed = data.get('maxspeed', None)
        if maxspeed:
            try:
                if isinstance(maxspeed, list):
                    maxspeed = maxspeed[0]
                # Handle different formats (e.g., "50", "50 km/h", "30 mph")
                if isinstance(maxspeed, str):
                    maxspeed = maxspeed.replace(' km/h', '').replace(' mph', '')
                    if 'mph' in str(data.get('maxspeed', '')):
                        speed_kmh = float(maxspeed) * 1.60934  # Convert mph to km/h
                    else:
                        speed_kmh = float(maxspeed)
                else:
                    speed_kmh = float(maxspeed)
            except (ValueError, TypeError):
                speed_kmh = DEFAULT_SPEEDS.get(highway_type, 40)
        else:
            speed_kmh = DEFAULT_SPEEDS.get(highway_type, 40)
        
        # Calculate time in seconds
        time_seconds = (length / 1000) / speed_kmh * 3600  # Convert to hours then to seconds
        
        # Get road quality
        quality = ROAD_QUALITY.get(highway_type, 0.5)
        
        # Store raw values
        data['distance'] = length
        data['time'] = time_seconds
        data['speed'] = speed_kmh
        data['quality'] = quality
        data['highway_type'] = highway_type
        
        # Track max values
        max_distance = max(max_distance, length)
        max_time = max(max_time, time_seconds)
    
    # Second pass: compute normalized composite cost
    for u, v, key, data in G.edges(keys=True, data=True):
        # Normalize components to [0, 1]
        norm_distance = data['distance'] / max_distance if max_distance > 0 else 0
        norm_time = data['time'] / max_time if max_time > 0 else 0
        norm_quality = 1 - data['quality']  # Invert so lower quality = higher cost
        
        # Composite cost (lower is better)
        cost = (distance_weight * norm_distance + 
                time_weight * norm_time + 
                quality_weight * norm_quality)
        
        data['cost'] = cost
        
        # Also store the actual distance/time for metrics
        data['weight'] = cost  # NetworkX uses 'weight' for shortest path algorithms
    
    print(f"Preprocessed {len(G.edges)} edges with cost attributes")
    print(f"Cost weights: distance={distance_weight:.2f}, time={time_weight:.2f}, quality={quality_weight:.2f}")
    
    return G


def get_random_node_pairs(G: nx.MultiDiGraph, 
                          n_pairs: int = 100,
                          min_distance: float = 1000,
                          max_distance: float = 10000,
                          seed: int = 42) -> List[Tuple[int, int]]:
    """
    Generate random start-goal node pairs for training/testing.
    
    Args:
        G: Road network graph
        n_pairs: Number of pairs to generate
        min_distance: Minimum straight-line distance between nodes (meters)
        max_distance: Maximum straight-line distance between nodes (meters)
        seed: Random seed for reproducibility
        
    Returns:
        List of (start_node, goal_node) tuples
    """
    np.random.seed(seed)
    nodes = list(G.nodes())
    pairs = []
    
    print(f"Generating {n_pairs} random node pairs...")
    
    attempts = 0
    max_attempts = n_pairs * 100
    
    while len(pairs) < n_pairs and attempts < max_attempts:
        attempts += 1
        
        # Sample two random nodes
        start, goal = np.random.choice(nodes, size=2, replace=False)
        
        # Check if there's a path between them
        if not nx.has_path(G, start, goal):
            continue
        
        # Calculate straight-line distance
        start_pos = (G.nodes[start]['y'], G.nodes[start]['x'])
        goal_pos = (G.nodes[goal]['y'], G.nodes[goal]['x'])
        
        # Haversine distance calculation
        from math import radians, cos, sin, asin, sqrt
        lat1, lon1 = radians(start_pos[0]), radians(start_pos[1])
        lat2, lon2 = radians(goal_pos[0]), radians(goal_pos[1])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        distance = 6371000 * c  # Earth radius in meters
        
        # Check distance constraints
        if min_distance <= distance <= max_distance:
            pairs.append((start, goal))
    
    if len(pairs) < n_pairs:
        print(f"Warning: Only generated {len(pairs)} pairs (requested {n_pairs})")
    else:
        print(f"Generated {len(pairs)} valid node pairs")
    
    return pairs


def create_small_graph(G: nx.MultiDiGraph, 
                       n_nodes: int = 500,
                       center_node: int = None,
                       seed: int = 42) -> nx.MultiDiGraph:
    """
    Create a smaller connected subgraph using BFS expansion.
    
    This ensures the subgraph stays connected by growing outward from
    a starting node using breadth-first search.
    
    Args:
        G: Full road network graph
        n_nodes: Target number of nodes in the subgraph
        center_node: Optional starting node (random if None)
        seed: Random seed for reproducibility
        
    Returns:
        Smaller connected subgraph
    """
    np.random.seed(seed)
    
    print(f"Creating subgraph with ~{n_nodes} nodes...")
    
    # Get the largest weakly connected component first
    largest_cc = max(nx.weakly_connected_components(G), key=len)
    G_connected = G.subgraph(largest_cc)
    
    print(f"Largest connected component has {len(G_connected.nodes)} nodes")
    
    # If requesting more nodes than available, use all
    if n_nodes >= len(G_connected.nodes):
        print(f"Using entire connected component ({len(G_connected.nodes)} nodes)")
        return G_connected.copy()
    
    # Pick random starting node
    if center_node is None or center_node not in G_connected:
        center_node = np.random.choice(list(G_connected.nodes()))
    
    # BFS expansion to maintain connectivity
    subgraph_nodes = set([center_node])
    frontier = set([center_node])
    
    while len(subgraph_nodes) < n_nodes and frontier:
        # Get all neighbors of current frontier
        new_frontier = set()
        for node in frontier:
            # Add neighbors (both incoming and outgoing for directed graph)
            for neighbor in G_connected.successors(node):
                if neighbor not in subgraph_nodes:
                    new_frontier.add(neighbor)
            for neighbor in G_connected.predecessors(node):
                if neighbor not in subgraph_nodes:
                    new_frontier.add(neighbor)
        
        # If we would exceed target, sample from new frontier
        if len(subgraph_nodes) + len(new_frontier) > n_nodes:
            remaining = n_nodes - len(subgraph_nodes)
            new_frontier = set(np.random.choice(list(new_frontier), 
                                               size=remaining, 
                                               replace=False))
        
        # Add new frontier to subgraph
        subgraph_nodes.update(new_frontier)
        frontier = new_frontier
    
    # Create the subgraph
    subgraph = G.subgraph(subgraph_nodes).copy()
    
    print(f"Created subgraph with {len(subgraph.nodes)} nodes and {len(subgraph.edges)} edges")
    
    return subgraph


def split_train_test(pairs: List[Tuple[int, int]], 
                     test_ratio: float = 0.2,
                     seed: int = 42) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Split node pairs into training and testing sets.
    
    Args:
        pairs: List of (start, goal) node pairs
        test_ratio: Ratio of pairs to use for testing
        seed: Random seed
        
    Returns:
        (train_pairs, test_pairs)
    """
    np.random.seed(seed)
    n_test = int(len(pairs) * test_ratio)
    
    # Shuffle pairs
    shuffled = pairs.copy()
    np.random.shuffle(shuffled)
    
    test_pairs = shuffled[:n_test]
    train_pairs = shuffled[n_test:]
    
    print(f"Split: {len(train_pairs)} training pairs, {len(test_pairs)} test pairs")
    
    return train_pairs, test_pairs


if __name__ == "__main__":
    # Test the data loader
    print("Testing data loader...")
    
    # Load graph
    G = load_quebec_graph()
    
    # Preprocess
    G = preprocess_graph(G)
    
    # Generate pairs
    pairs = get_random_node_pairs(G, n_pairs=50)
    train_pairs, test_pairs = split_train_test(pairs)
    
    print("\nData loader test complete!")
    print(f"Graph: {len(G.nodes)} nodes, {len(G.edges)} edges")
    print(f"Pairs: {len(train_pairs)} train, {len(test_pairs)} test")
