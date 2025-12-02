"""
Routing Environment for Reinforcement Learning

This module implements the RL environment for route optimization.
The agent navigates through the Quebec City road network to find optimal paths.
"""

import networkx as nx
import numpy as np
from typing import Tuple, List, Dict, Any, Optional


class RoutingEnv:
    """
    Reinforcement Learning environment for route optimization.
    
    State: (current_node, goal_node, distance_to_goal, current_road_features)
    Actions: Available neighbor nodes
    Reward: Negative cost with penalties for loops and bonuses for progress
    """
    
    def __init__(self, graph: nx.MultiDiGraph, max_steps: int = 500):
        """
        Initialize the routing environment.
        
        Args:
            graph: Road network graph with cost attributes
            max_steps: Maximum steps per episode before termination
        """
        self.graph = graph
        self.max_steps = max_steps
        
        # Create node to index mapping for state representation
        self.nodes = list(graph.nodes())
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        self.n_nodes = len(self.nodes)
        
        # Episode state
        self.current_node = None
        self.goal_node = None
        self.start_node = None
        self.visited_nodes = set()
        self.path = []
        self.total_cost = 0.0
        self.steps = 0
        
        # Precompute node positions for distance calculations
        self.node_positions = {
            node: (data['y'], data['x']) 
            for node, data in graph.nodes(data=True)
        }
        
    def reset(self, start: int, goal: int) -> Dict[str, Any]:
        """
        Reset the environment for a new episode.
        
        Args:
            start: Starting node ID
            goal: Goal node ID
            
        Returns:
            Initial state dictionary
        """
        self.current_node = start
        self.goal_node = goal
        self.start_node = start
        self.visited_nodes = {start}
        self.path = [start]
        self.total_cost = 0.0
        self.steps = 0
        
        return self.get_state()
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state representation.
        
        Returns:
            Dictionary containing:
                - current_node: Current node ID
                - goal_node: Goal node ID
                - current_idx: Index of current node
                - goal_idx: Index of goal node
                - distance_to_goal: Straight-line distance to goal
                - steps: Number of steps taken
                - visited_count: Number of unique nodes visited
        """
        # Calculate straight-line distance to goal
        current_pos = self.node_positions[self.current_node]
        goal_pos = self.node_positions[self.goal_node]
        
        # Simple Euclidean distance (approximation)
        distance_to_goal = np.sqrt(
            (current_pos[0] - goal_pos[0])**2 + 
            (current_pos[1] - goal_pos[1])**2
        )
        
        return {
            'current_node': self.current_node,
            'goal_node': self.goal_node,
            'current_idx': self.node_to_idx[self.current_node],
            'goal_idx': self.node_to_idx[self.goal_node],
            'distance_to_goal': distance_to_goal,
            'steps': self.steps,
            'visited_count': len(self.visited_nodes),
        }
    
    def get_available_actions(self) -> List[int]:
        """
        Get list of available actions (neighbor node indices) from current state.
        
        Returns:
            List of neighbor node indices (not IDs)
        """
        # Get all outgoing edges from current node
        neighbors = []
        for _, neighbor in self.graph.out_edges(self.current_node):
            # Convert node ID to index
            neighbors.append(self.node_to_idx[neighbor])
        
        return neighbors if neighbors else [self.node_to_idx[self.current_node]]  # Stay in place if no neighbors
    
    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute an action (move to a neighbor node).
        
        Args:
            action: Next node INDEX to move to (not node ID)
            
        Returns:
            (next_state, reward, done, info)
        """
        self.steps += 1
        
        # Check if action is valid
        available_actions = self.get_available_actions()
        if action not in available_actions:
            # Invalid action - stay in place with penalty
            return self.get_state(), -10.0, False, {'invalid_action': True}
        
        # Convert action index to node ID
        action_node_id = self.nodes[action]
        
        # Calculate cost of this move
        edge_cost = self._get_edge_cost(self.current_node, action_node_id)
        
        # Calculate previous distance to goal
        prev_distance = self.get_state()['distance_to_goal']
        
        # Move to next node
        prev_node = self.current_node
        self.current_node = action_node_id
        self.path.append(action_node_id)
        self.total_cost += edge_cost
        
        # Calculate new distance to goal
        new_distance = self.get_state()['distance_to_goal']
        
        # Calculate reward
        reward = self._calculate_reward(
            edge_cost=edge_cost,
            prev_distance=prev_distance,
            new_distance=new_distance,
            action=action_node_id,
            prev_node=prev_node
        )
        
        # Check if goal reached
        done = (self.current_node == self.goal_node)
        
        # Check if max steps exceeded
        if self.steps >= self.max_steps:
            done = True
            reward -= 50.0  # Penalty for timeout
        
        # Track visited nodes
        self.visited_nodes.add(self.current_node)
        
        # Info dictionary
        info = {
            'path_length': len(self.path),
            'total_cost': self.total_cost,
            'goal_reached': self.current_node == self.goal_node,
            'edge_cost': edge_cost,
        }
        
        return self.get_state(), reward, done, info
    
    def _get_edge_cost(self, u: int, v: int) -> float:
        """Get the cost of moving from node u to node v."""
        # Handle MultiDiGraph - there might be multiple edges
        if self.graph.has_edge(u, v):
            # Get the edge with minimum cost
            min_cost = float('inf')
            for key in self.graph[u][v]:
                cost = self.graph[u][v][key].get('cost', 1.0)
                min_cost = min(min_cost, cost)
            return min_cost
        return 1.0  # Default cost if edge doesn't exist
    
    def _calculate_reward(self, 
                         edge_cost: float,
                         prev_distance: float,
                         new_distance: float,
                         action: int,
                         prev_node: int) -> float:
        """
        Calculate reward for the current step.
        
        Reward components:
        - Negative edge cost (lower cost = higher reward)
        - Progress bonus (moving closer to goal)
        - Loop penalty (revisiting nodes)
        - Goal bonus (reaching the goal)
        """
        # Base reward: negative of edge cost (scaled)
        reward = -edge_cost * 10.0
        
        # Progress reward: bonus for getting closer to goal
        progress = prev_distance - new_distance
        reward += progress * 50.0  # Scale progress reward
        
        # Loop penalty: penalize revisiting nodes
        if action in self.visited_nodes and action != self.goal_node:
            reward -= 5.0
        
        # Goal bonus: large reward for reaching goal
        if action == self.goal_node:
            reward += 100.0
            # Additional bonus for shorter paths
            reward += max(0, 50.0 - self.steps * 0.5)
        
        return reward
    
    def get_path_metrics(self) -> Dict[str, float]:
        """
        Calculate metrics for the current path.
        
        Returns:
            Dictionary with distance, time, and cost metrics
        """
        total_distance = 0.0
        total_time = 0.0
        total_cost = 0.0
        
        for i in range(len(self.path) - 1):
            u, v = self.path[i], self.path[i + 1]
            
            if self.graph.has_edge(u, v):
                # Get edge with minimum cost
                min_cost = float('inf')
                best_edge = None
                
                for key in self.graph[u][v]:
                    cost = self.graph[u][v][key].get('cost', 1.0)
                    if cost < min_cost:
                        min_cost = cost
                        best_edge = self.graph[u][v][key]
                
                if best_edge:
                    total_distance += best_edge.get('distance', 0)
                    total_time += best_edge.get('time', 0)
                    total_cost += best_edge.get('cost', 0)
        
        return {
            'distance_m': total_distance,
            'time_s': total_time,
            'time_min': total_time / 60.0,
            'cost': total_cost,
            'num_nodes': len(self.path),
        }


if __name__ == "__main__":
    # Test the environment
    print("Testing routing environment...")
    
    from data_loader import load_quebec_graph, preprocess_graph
    
    # Load and preprocess graph
    G = load_quebec_graph()
    G = preprocess_graph(G)
    
    # Create environment
    env = RoutingEnv(G)
    
    # Get two random connected nodes
    nodes = list(G.nodes())
    start, goal = nodes[0], nodes[100]
    
    # Reset environment
    state = env.reset(start, goal)
    print(f"\nStarting episode: {start} -> {goal}")
    print(f"Initial state: {state}")
    
    # Take a few random steps
    for i in range(5):
        actions = env.get_available_actions()
        action = np.random.choice(actions)
        
        next_state, reward, done, info = env.step(action)
        print(f"\nStep {i+1}:")
        print(f"  Action: {action}")
        print(f"  Reward: {reward:.2f}")
        print(f"  Done: {done}")
        print(f"  Info: {info}")
        
        if done:
            break
    
    print("\nEnvironment test complete!")
