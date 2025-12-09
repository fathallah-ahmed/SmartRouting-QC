# SmartRouting-QC: Comparing Deep RL vs Classical Algorithms for Route Optimization

**A comparative study demonstrating why classical algorithms (A*) outperform Deep Reinforcement Learning (DQN) for shortest-path routing problems.**

## 🎯 Project Overview

This project implements and compares two approaches for route optimization on Quebec City's real road network:

1. **Deep Q-Learning (DQN)** - Neural network-based reinforcement learning
2. **A* Algorithm** - Classical graph search with heuristic

### Key Finding: A* Dominates

| Metric | DQN (200 nodes, 1000 episodes) | A* (5000+ nodes, instant) |
|--------|-------------------------------|---------------------------|
| **Success Rate (Training)** | 19% | 100% |
| **Success Rate (Test)** | 0% | 100% |
| **Training Time** | ~26 minutes | 0 seconds |
| **Path Quality** | Suboptimal | Guaranteed optimal |
| **Scalability** | Struggles at 200+ nodes | Works on 11,000+ nodes |

## 📋 Requirements

```bash
Python 3.8+
See requirements.txt for dependencies
```

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
├── data/                       # Cached OSM data (auto-generated)
├── models/                     # Trained DQN models
├── results/                    # Evaluation results and figures
│   ├── logs/                  # Training logs and metrics
│   └── evaluation_results.csv # Performance comparison data
├── src/
│   ├── data_loader.py         # OSM data loading, preprocessing, subgraph creation
│   ├── env_routing.py         # RL environment
│   ├── agent_rl.py            # DQN agent implementation
│   ├── astar_routing.py       # A* algorithm (NEW)
│   ├── baseline.py            # Dijkstra baseline
│   ├── train.py               # DQN training script
│   ├── evaluate.py            # Evaluation and benchmarking
│   └── visualize.py           # Visualization utilities
├── compare_algorithms.py       # A* vs DQN comparison (NEW)
├── requirements.txt
└── README.md
```

## 🎓 Usage

### Option 1: Use A* (Recommended) ✅

Test A* algorithm on Quebec City's road network:

```bash
python src/astar_routing.py
```

**Output:**
```
[SUCCESS] Path found!
   Hops: 132
   Cost: 12.61
   Distance: 24876m
```

**Why A*?**
- ✅ 100% success rate
- ✅ Guaranteed optimal paths
- ✅ No training required
- ✅ Industry standard (Google Maps, navigation systems)

### Option 2: Try DQN (Educational) 📚

Train DQN agent (for comparison/learning purposes):

```bash
# Train on small 200-node graph
python src/train.py --small-graph-nodes 200 --episodes 1000
```

**Expected results:**
- Training success: ~19%
- Test success: ~0%
- Training time: 25-30 minutes

Evaluate the trained model:

```bash
python src/evaluate.py --model-path models/final_model.pth \
    --use-small-graph --small-graph-nodes 200 \
    --distance-weight 1.0 --time-weight 0.0 --quality-weight 0.0
```

### Option 3: Compare Both Algorithms

Run comprehensive comparison:

```bash
# First train DQN
python src/train.py --small-graph-nodes 200

# Then compare
python compare_algorithms.py
```

**Sample output:**
```
============================================================
COMPARISON: A* vs DQN
============================================================

DQN (after 1000 episodes):
  Success Rate: 9.0%
  Avg Reward: -3508.6

A* (no training needed):
  Success Rate: 100.0%
  Avg Cost: 2.41

WINNER: A* 🎯
```

## 📊 Experimental Results

### DQN Performance Analysis

**Training Configuration:**
- Graph size: 200 nodes, 490 edges
- Episodes: 1,000
- Learning rate: 0.00005
- Reward function: Distance-based progress + goal bonus

**Results:**

| Graph Size | Training Success | Test Success | Training Time |
|------------|-----------------|--------------|---------------|
| 200 nodes | 19% | **0%** | 26 minutes |
| 500 nodes | 9% | N/A | 29 minutes |
| 5000 nodes | 3% | N/A | Failed (loss explosion) |

**Key Issues Observed:**
1. **Zero generalization**: 19% training success → 0% test success
2. **Loss explosion**: Loss increased to 2-5 million (should decrease)
3. **Poor scalability**: Performance degraded with larger graphs
4. **No convergence**: Agent couldn't learn stable policies

### A* Performance

**Tested on:**
- Full graph: 11,602 nodes, 31,552 edges ✅
- 5000-node subgraph ✅
- 200-node subgraph ✅

**Results:**
- Success rate: **100%** on all graph sizes
- Path quality: **Optimal** (guaranteed)
- Computation time: **Instant** (<1 second per route)

## 🧠 Why DQN Failed

### 1. **Problem Mismatch**
Routing is a **solved problem** with optimal classical algorithms. DQN is designed for:
- Games (Atari, Go) where optimal strategy is unknown
- Robotics where environment dynamics are complex
- Problems without closed-form solutions

### 2. **State Space Explosion**
- Even 200 nodes = 200² = 40,000 possible state-action pairs
- DQN's neural network can't efficiently learn this mapping
- Classical algorithms exploit graph structure directly

### 3. **Sparse Rewards**
- Agent only gets meaningful feedback when reaching goal (~19% of time)
- Insufficient learning signal for gradient descent
- A* uses admissible heuristic for every decision

### 4. **Overfitting**
- 19% success on training pairs
- 0% success on test pairs
- Agent memorized specific routes, didn't learn routing principles

## � Implementation Details

### DQN Architecture

- **Input**: 9-dimensional state vector
  - Current/goal node indices
  - Distance to goal
  - Steps taken, visited count
  - Normalized lat/lon coordinates
- **Hidden Layers**: 256 → 256 → 128 neurons (ReLU)
- **Output**: Q-values for each node
- **Training**: Experience replay, target network, ε-greedy exploration

### A* Algorithm

- **Heuristic**: Haversine distance (straight-line GPS distance)
- **Admissibility**: Never overestimates true cost → guaranteed optimality
- **Data Structure**: Priority queue (min-heap)
- **Complexity**: O(E log V) where E = edges, V = nodes

## � Cost Function

Routes can be optimized for different criteria:

```python
Cost = w₁ × distance + w₂ × time + w₃ × road_quality
```

Where:
- **Distance**: Edge length from OSM (meters)
- **Time**: Estimated using distance/speed
- **Road Quality**: Penalty/bonus based on highway type

**Examples:**
```bash
# Optimize for distance only
python src/train.py --distance-weight 1.0 --time-weight 0.0 --quality-weight 0.0

# Optimize for time
python src/train.py --distance-weight 0.2 --time-weight 0.7 --quality-weight 0.1
```

## 🎓 Key Learnings

### When to Use DQN:
- ✅ Complex, high-dimensional continuous control (robotics)
- ✅ Games with unknown optimal strategies
- ✅ Problems where classical algorithms don't exist
- ✅ When approximate solutions are acceptable

### When to Use Classical Algorithms (A*, Dijkstra):
- ✅ **Shortest path problems** (like this project)
- ✅ When optimal solutions exist and are required
- ✅ When interpretability matters
- ✅ When training time/data is limited
- ✅ Production systems requiring reliability

### Project Takeaway:
**Not every problem needs deep learning.** This project demonstrates that understanding your problem domain and choosing the right algorithm is more important than using the latest ML techniques.

## 🛠️ Advanced DQN Training (For Research)

If you still want to experiment with DQN:

```bash
# Smallest viable graph
python src/train.py --small-graph-nodes 100 --episodes 2000

# More training
python src/train.py --small-graph-nodes 200 --episodes 5000

# Tune hyperparameters
python src/train.py \
    --small-graph-nodes 200 \
    --learning-rate 0.00003 \
    --epsilon-decay 0.9998 \
    --episodes 3000
```

**Note**: Even with tuning, DQN is unlikely to match A*'s performance.

## 📊 Visualization

Plot training metrics:

```python
from src.visualize import plot_training_metrics
plot_training_metrics('results/logs/training_metrics.json', 'results/figures')
```

Compare evaluation results:

```python
from src.visualize import plot_evaluation_metrics
plot_evaluation_metrics('results/evaluation_results.csv', 'results/figures')
```

## 🤝 Contributing

Contributions welcome! Potential improvements:
- Compare against other RL algorithms (PPO, SAC, etc.)
- Test Graph Neural Networks (GNN)
- Add more classical baselines (Bidirectional Dijkstra, etc.)
- Improve reward shaping for DQN

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- OpenStreetMap for providing road network data
- OSMnx library for easy OSM data access
- PyTorch for deep learning framework
- Quebec City for being a great test case!

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

## 💡 Conclusion

This project serves as an educational demonstration that:

1. **Classical algorithms remain relevant** - Don't overlook proven methods
2. **Problem selection matters** - DQN excels in some domains, fails in others
3. **Benchmarking is crucial** - Always compare against simple baselines
4. **Understand your problem** - Routing = solved problem → use A*/Dijkstra

**For production route optimization: Use A\***

**For learning about DRL: This project shows both successes and limitations**

---

**Note**: First run will download Quebec City's road network from OpenStreetMap (~2-3 minutes). Subsequent runs use cached data.
