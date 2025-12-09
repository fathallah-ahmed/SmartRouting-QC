# Training Guide

## Quick Start

Train the DQN agent on a 10,000-node Quebec City subgraph:

```bash
python src/train.py
```

This uses optimized defaults for the 10k-node subgraph, providing an excellent balance between training speed and network realism.

## Configuration

### Default Settings (10,000 nodes)

- **Graph size**: 10,000 nodes (~1/3 of full Quebec City)
- **Episodes**: 1,000
- **Node pairs**: 200
- **Max steps**: 500
- **Memory usage**: ~500-800 MB (vs 2-4 GB for full graph)

### Custom Graph Size

To use a different subgraph size:

```bash
# Smaller for faster training
python src/train.py --small-graph-nodes 5000

# Larger for more realism
python src/train.py --small-graph-nodes 20000

# Use full graph (not recommended - very slow)
python src/train.py --use-small-graph false
```

## Training Parameters

Common parameters you might want to adjust:

```bash
python src/train.py \
    --episodes 2000 \
    --learning-rate 0.0003 \
    --epsilon-decay 0.999 \
    --hidden-dim 256
```

See `python src/train.py --help` for all options.

## Output

- **Models**: Saved to `models/` directory
- **Logs**: Saved to `results/logs/`
- **Checkpoints**: Every 500 episodes
- **Training curves**: Generated automatically

## Tips

- Default 10k nodes provides good training speed and realistic network
- Reduce to 5k nodes for rapid iteration during development
- Increase to 15-20k nodes for final training runs
