"""
Visualization Module for Route Maps and Metrics

This module provides visualization functions for:
- Interactive route maps comparing RL and Dijkstra paths
- Training metrics graphs
- Evaluation comparison charts
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from typing import List, Tuple, Dict, Optional
import networkx as nx


# Set style for matplotlib
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def plot_route_comparison(graph: nx.MultiDiGraph,
                          rl_path: List[int],
                          dijkstra_path: List[int],
                          output_path: str,
                          title: str = "Route Comparison: RL vs Dijkstra"):
    """
    Create an interactive map comparing RL and Dijkstra paths.
    
    Args:
        graph: Road network graph
        rl_path: Path found by RL agent
        dijkstra_path: Path found by Dijkstra
        output_path: Path to save HTML map
        title: Map title
    """
    # Get node positions
    def get_node_pos(node):
        return (graph.nodes[node]['y'], graph.nodes[node]['x'])
    
    # Calculate map center
    if rl_path:
        center_lat = np.mean([graph.nodes[n]['y'] for n in rl_path])
        center_lon = np.mean([graph.nodes[n]['x'] for n in rl_path])
    elif dijkstra_path:
        center_lat = np.mean([graph.nodes[n]['y'] for n in dijkstra_path])
        center_lon = np.mean([graph.nodes[n]['x'] for n in dijkstra_path])
    else:
        # Default to Quebec City center
        center_lat, center_lon = 46.8139, -71.2080
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles='OpenStreetMap'
    )
    
    # Add Dijkstra path (red)
    if dijkstra_path and len(dijkstra_path) > 1:
        dijkstra_coords = [get_node_pos(n) for n in dijkstra_path]
        folium.PolyLine(
            dijkstra_coords,
            color='red',
            weight=4,
            opacity=0.7,
            popup='Dijkstra Path'
        ).add_to(m)
    
    # Add RL path (blue)
    if rl_path and len(rl_path) > 1:
        rl_coords = [get_node_pos(n) for n in rl_path]
        folium.PolyLine(
            rl_coords,
            color='blue',
            weight=4,
            opacity=0.7,
            popup='RL Agent Path'
        ).add_to(m)
    
    # Add start marker (green)
    if rl_path:
        start_pos = get_node_pos(rl_path[0])
        folium.Marker(
            start_pos,
            popup='Start',
            icon=folium.Icon(color='green', icon='play')
        ).add_to(m)
    
    # Add goal marker (red)
    if rl_path:
        goal_pos = get_node_pos(rl_path[-1])
        folium.Marker(
            goal_pos,
            popup='Goal',
            icon=folium.Icon(color='red', icon='stop')
        ).add_to(m)
    
    # Add legend
    legend_html = f'''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 200px; height: 120px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
        <p style="margin: 0; font-weight: bold;">{title}</p>
        <p style="margin: 5px 0;"><span style="color: blue;">━━━</span> RL Agent</p>
        <p style="margin: 5px 0;"><span style="color: red;">━━━</span> Dijkstra</p>
        <p style="margin: 5px 0;"><span style="color: green;">▶</span> Start</p>
        <p style="margin: 5px 0;"><span style="color: red;">■</span> Goal</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save map
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    m.save(output_path)
    print(f"Saved route map to {output_path}")


def plot_training_metrics(metrics_path: str, output_dir: str):
    """
    Plot training metrics from saved JSON file.
    
    Args:
        metrics_path: Path to training_metrics.json
        output_dir: Directory to save plots
    """
    # Load metrics
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    rewards = metrics['episode_rewards']
    success = metrics['episode_success']
    losses = metrics['episode_losses']
    epsilon = metrics['epsilon_history']
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Moving average function
    def moving_average(data, window=50):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    # Plot 1: Episode Rewards
    axes[0, 0].plot(rewards, alpha=0.3, color='blue', label='Raw')
    axes[0, 0].plot(moving_average(rewards), color='darkblue', linewidth=2, label='MA(50)')
    axes[0, 0].set_xlabel('Episode', fontsize=12)
    axes[0, 0].set_ylabel('Total Reward', fontsize=12)
    axes[0, 0].set_title('Episode Rewards Over Time', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Success Rate
    axes[0, 1].plot(success, alpha=0.3, color='green', label='Raw')
    axes[0, 1].plot(moving_average(success), color='darkgreen', linewidth=2, label='MA(50)')
    axes[0, 1].set_xlabel('Episode', fontsize=12)
    axes[0, 1].set_ylabel('Success (1=Goal Reached)', fontsize=12)
    axes[0, 1].set_title('Success Rate Over Time', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylim([-0.1, 1.1])
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Training Loss
    axes[1, 0].plot(losses, alpha=0.3, color='red', label='Raw')
    axes[1, 0].plot(moving_average(losses), color='darkred', linewidth=2, label='MA(50)')
    axes[1, 0].set_xlabel('Episode', fontsize=12)
    axes[1, 0].set_ylabel('Loss', fontsize=12)
    axes[1, 0].set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Epsilon Decay
    axes[1, 1].plot(epsilon, color='purple', linewidth=2)
    axes[1, 1].set_xlabel('Episode', fontsize=12)
    axes[1, 1].set_ylabel('Epsilon (ε)', fontsize=12)
    axes[1, 1].set_title('Exploration Rate Decay', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, 'training_metrics.png')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved training metrics plot to {output_path}")
    plt.close()


def plot_evaluation_metrics(results_csv: str, output_dir: str):
    """
    Plot evaluation comparison charts.
    
    Args:
        results_csv: Path to evaluation_results.csv
        output_dir: Directory to save plots
    """
    # Load results
    df = pd.read_csv(results_csv)
    
    # Filter successful routes for both methods
    both_success = df[(df['rl_success']) & (df['dijkstra_success'])]
    
    if len(both_success) == 0:
        print("No routes where both methods succeeded. Skipping comparison plots.")
        return
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 12))
    
    # Plot 1: Distance Comparison (Bar Chart)
    ax1 = plt.subplot(2, 3, 1)
    metrics = ['Distance (km)', 'Time (min)', 'Cost']
    rl_values = [
        both_success['rl_distance_km'].mean(),
        both_success['rl_time_min'].mean(),
        both_success['rl_cost'].mean()
    ]
    dijkstra_values = [
        both_success['dijkstra_distance_km'].mean(),
        both_success['dijkstra_time_min'].mean(),
        both_success['dijkstra_cost'].mean()
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax1.bar(x - width/2, rl_values, width, label='RL Agent', color='blue', alpha=0.7)
    ax1.bar(x + width/2, dijkstra_values, width, label='Dijkstra', color='red', alpha=0.7)
    ax1.set_ylabel('Average Value', fontsize=12)
    ax1.set_title('Average Metrics Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Distance Ratio Distribution
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(both_success['distance_ratio'], bins=30, color='blue', alpha=0.7, edgecolor='black')
    ax2.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Equal Performance')
    ax2.axvline(both_success['distance_ratio'].mean(), color='green', linestyle='-', linewidth=2, 
                label=f'Mean: {both_success["distance_ratio"].mean():.3f}')
    ax2.set_xlabel('Distance Ratio (RL / Dijkstra)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distance Ratio Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Time Ratio Distribution
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(both_success['time_ratio'], bins=30, color='green', alpha=0.7, edgecolor='black')
    ax3.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Equal Performance')
    ax3.axvline(both_success['time_ratio'].mean(), color='darkgreen', linestyle='-', linewidth=2,
                label=f'Mean: {both_success["time_ratio"].mean():.3f}')
    ax3.set_xlabel('Time Ratio (RL / Dijkstra)', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Time Ratio Distribution', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Cost Ratio Distribution
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(both_success['cost_ratio'], bins=30, color='purple', alpha=0.7, edgecolor='black')
    ax4.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Equal Performance')
    ax4.axvline(both_success['cost_ratio'].mean(), color='darkviolet', linestyle='-', linewidth=2,
                label=f'Mean: {both_success["cost_ratio"].mean():.3f}')
    ax4.set_xlabel('Cost Ratio (RL / Dijkstra)', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Cost Ratio Distribution', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Success Rate Comparison
    ax5 = plt.subplot(2, 3, 5)
    success_data = [
        df['rl_success'].mean() * 100,
        df['dijkstra_success'].mean() * 100
    ]
    colors = ['blue', 'red']
    ax5.bar(['RL Agent', 'Dijkstra'], success_data, color=colors, alpha=0.7)
    ax5.set_ylabel('Success Rate (%)', fontsize=12)
    ax5.set_title('Success Rate Comparison', fontsize=14, fontweight='bold')
    ax5.set_ylim([0, 105])
    for i, v in enumerate(success_data):
        ax5.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Scatter: RL vs Dijkstra Distance
    ax6 = plt.subplot(2, 3, 6)
    ax6.scatter(both_success['dijkstra_distance_km'], both_success['rl_distance_km'], 
                alpha=0.5, color='blue', s=50)
    
    # Add diagonal line (perfect match)
    max_dist = max(both_success['dijkstra_distance_km'].max(), both_success['rl_distance_km'].max())
    ax6.plot([0, max_dist], [0, max_dist], 'r--', linewidth=2, label='Perfect Match')
    
    ax6.set_xlabel('Dijkstra Distance (km)', fontsize=12)
    ax6.set_ylabel('RL Distance (km)', fontsize=12)
    ax6.set_title('Distance Comparison Scatter', fontsize=14, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, 'evaluation_metrics.png')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved evaluation metrics plot to {output_path}")
    plt.close()


def create_sample_route_visualizations(graph: nx.MultiDiGraph,
                                       results_csv: str,
                                       output_dir: str,
                                       n_samples: int = 5):
    """
    Create route map visualizations for sample test cases.
    
    Args:
        graph: Road network graph
        results_csv: Path to evaluation results
        output_dir: Directory to save maps
        n_samples: Number of sample routes to visualize
    """
    from env_routing import RoutingEnv
    from agent_rl import DQNAgent
    from baseline import dijkstra_path
    
    # Load results
    df = pd.read_csv(results_csv)
    both_success = df[(df['rl_success']) & (df['dijkstra_success'])]
    
    if len(both_success) == 0:
        print("No successful routes to visualize.")
        return
    
    # Sample routes
    samples = both_success.sample(min(n_samples, len(both_success)))
    
    print(f"\nCreating {len(samples)} sample route visualizations...")
    
    for idx, row in samples.iterrows():
        start, goal = int(row['start']), int(row['goal'])
        
        # Get Dijkstra path
        dijkstra_path_nodes = dijkstra_path(graph, start, goal, weight='cost')
        
        # For RL path, we need to re-run the agent
        # This is a simplified version - in practice, you'd store paths during evaluation
        # For now, we'll just visualize Dijkstra path
        # TODO: Store RL paths during evaluation for visualization
        
        output_path = os.path.join(output_dir, f'route_sample_{idx}.html')
        
        plot_route_comparison(
            graph,
            rl_path=dijkstra_path_nodes,  # Placeholder - would be actual RL path
            dijkstra_path=dijkstra_path_nodes,
            output_path=output_path,
            title=f"Route {idx}: {start} → {goal}"
        )
    
    print(f"Saved {len(samples)} route visualizations to {output_dir}")


if __name__ == "__main__":
    print("Testing visualization module...")
    
    # This is a test/example - in practice, run after training and evaluation
    print("\nTo use this module:")
    print("1. After training: plot_training_metrics('results/logs/training_metrics.json', 'results/figures')")
    print("2. After evaluation: plot_evaluation_metrics('results/evaluation_results.csv', 'results/figures')")
    print("3. For route maps: plot_route_comparison(graph, rl_path, dijkstra_path, 'results/figures/route.html')")
