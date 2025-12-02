# SmartRouting-QC: Route Optimization with Reinforcement Learning

A Deep Q-Learning (DQN) agent for optimizing routes in Quebec City using OpenStreetMap data, with comparison against Dijkstra's algorithm baseline.

## 🎯 Project Overview

This project implements a Reinforcement Learning solution for route optimization that:
- Uses **Deep Q-Learning (DQN)** to learn optimal routing policies
- Operates on real-world Quebec City road network from OpenStreetMap
- Optimizes for multiple criteria: distance, time, and road quality
- Compares performance against Dijkstra's algorithm baseline
- Provides comprehensive evaluation metrics and visualizations

### Why DQN?

**DQN was chosen over tabular Q-Learning** because:
- Quebec City's road network has thousands of nodes (state space explosion with tabular methods)
- Neural networks can generalize across similar road configurations
- Handles continuous state features (distance to goal, road attributes)
- More scalable for real-world routing applications

## 📋 Requirements

- Python 3.8+
- See `requirements.txt` for dependencies

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/SmartRouting-QC.git
cd SmartRouting-QC

# Install dependencies
pip install -r requirements.txt
```

## 📁 Project Structure

```
SmartRouting-QC/
├── data/                    # Cached OSM data (auto-generated)
├── models/                  # Trained models (auto-generated)
├── results/                 # Evaluation results and figures
│   ├── logs/               # Training logs and metrics
│   └── figures/            # Visualization outputs
├── src/
│   ├── data_loader.py      # OSM data loading and preprocessing
│   ├── env_routing.py      # RL environment
│   ├── agent_rl.py         # DQN agent implementation
│   ├── baseline.py         # Dijkstra baseline
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation and benchmarking
│   └── visualize.py        # Visualization utilities
├── requirements.txt
└── README.md
```

## 🎓 Usage

### 1. Training the DQN Agent

Train the agent on Quebec City's road network:

```bash
python src/train.py --episodes 1000 --n-pairs 200
```

**Key arguments:**
- `--episodes`: Number of training episodes (default: 1000)
- `--n-pairs`: Number of origin-destination pairs to generate (default: 200)
- `--test-ratio`: Ratio of pairs for testing (default: 0.2)
- `--distance-weight`: Weight for distance in cost function (default: 0.4)
- `--time-weight`: Weight for time in cost function (default: 0.4)
- `--quality-weight`: Weight for road quality (default: 0.2)
- `--learning-rate`: DQN learning rate (default: 0.001)
- `--epsilon-decay`: Exploration decay rate (default: 0.995)

**Output:**
- Trained model saved to `models/final_model.pth`
- Training metrics saved to `results/logs/training_metrics.json`
- Training curves plot saved to `results/logs/training_curves.png`

### 2. Evaluating the Agent

Evaluate the trained agent against Dijkstra baseline on test data:

```bash
python src/evaluate.py --model-path models/final_model.pth
```

**Output:**
- Detailed results CSV: `results/evaluation_results.csv`
- Summary statistics: `results/evaluation_results_summary.json`
- Console output with performance comparison

### 3. Visualizing Results

Generate visualizations after training and evaluation:

```python
from src.visualize import plot_training_metrics, plot_evaluation_metrics

# Plot training metrics
plot_training_metrics('results/logs/training_metrics.json', 'results/figures')

# Plot evaluation comparison
plot_evaluation_metrics('results/evaluation_results.csv', 'results/figures')
```

**Visualization outputs:**
- `training_metrics.png`: Training progress (rewards, success rate, loss, epsilon)
- `evaluation_metrics.png`: Performance comparison charts
- Interactive route maps (HTML) comparing RL vs Dijkstra paths

## 📊 Cost Function

The routing cost combines three factors:

```
Cost = w₁ × distance + w₂ × time + w₃ × road_quality
```

Where:
- **Distance**: Edge length from OSM (meters)
- **Time**: Estimated using distance/speed (from maxspeed or road type defaults)
- **Road Quality**: Penalty/bonus based on highway type (motorway > primary > residential)

Weights are configurable via command-line arguments.

## 🧠 RL Environment

**State**: 
- Current node index
- Goal node index
- Straight-line distance to goal
- Number of steps taken
- Number of unique nodes visited

**Actions**: 
- Available neighbor nodes (dynamic action space)

**Reward**:
- Negative edge cost (lower cost = higher reward)
- Progress bonus (moving closer to goal)
- Loop penalty (revisiting nodes)
- Goal bonus (reaching destination)

## 📈 Evaluation Metrics

The evaluation compares RL agent vs Dijkstra on:
- **Success Rate**: Percentage of routes where goal was reached
- **Distance**: Total path distance (km)
- **Time**: Estimated travel time (minutes)
- **Cost**: Composite cost score
- **Performance Ratios**: RL metrics / Dijkstra metrics

Ratios < 1.0 indicate RL outperforms Dijkstra; > 1.0 indicates Dijkstra is better.

## 🔬 Example Results

After training for 1000 episodes:

```
Success Rates:
  RL Agent:  85.0% (34/40)
  Dijkstra:  100.0% (40/40)

Performance Ratios (RL / Dijkstra):
  Distance:  1.05 ± 0.12
  Time:      1.04 ± 0.11
  Cost:      1.03 ± 0.10

(Ratio < 1.0 means RL is better, > 1.0 means Dijkstra is better)
```

## 🛠️ Advanced Usage

### Custom Cost Weights

Optimize for different criteria:

```bash
# Optimize primarily for time
python src/train.py --distance-weight 0.2 --time-weight 0.7 --quality-weight 0.1

# Optimize primarily for distance
python src/train.py --distance-weight 0.8 --time-weight 0.1 --quality-weight 0.1
```

### Longer Training

For better performance, train for more episodes:

```bash
python src/train.py --episodes 5000 --checkpoint-freq 1000
```

### Testing Individual Modules

Each module can be tested independently:

```bash
# Test data loader
python src/data_loader.py

# Test environment
python src/env_routing.py

# Test agent
python src/agent_rl.py

# Test baseline
python src/baseline.py
```

## 📝 Implementation Details

### DQN Architecture

- **Input**: 5-dimensional state vector
- **Hidden Layers**: 256 → 256 → 128 neurons with ReLU activation
- **Output**: Q-values for each node (action)
- **Optimizer**: Adam with learning rate 0.001
- **Loss**: Mean Squared Error (MSE)

### Training Features

- Experience replay buffer (capacity: 10,000)
- Target network for stable Q-value targets
- ε-greedy exploration with decay
- Gradient clipping for stability
- Periodic checkpointing

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- OpenStreetMap for providing road network data
- OSMnx library for easy OSM data access
- Quebec City for being a great test case!

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: First run will download Quebec City's road network from OpenStreetMap (~few minutes). Subsequent runs will use cached data.
