# Spinformer

Research codebase for the Spinformer architecture — exploring quaternion-transformer hybrids.

## Structure

```
model/              # Core model components
  components.py     # RMSNorm, RoPE, SwiGLU MLP
  attention.py      # Multi-Head Attention (GQA + RoPE + Flash)
  transformer.py    # Transformer (baseline decoder-only LM)

ablation_suite/     # Algorithmic task benchmarks
  config.py         # Task, Model, Training configs
  train.py          # Training loop with AMP + cosine LR
  evaluate.py       # ID + OOD length generalization eval
  metrics.py        # MetricsTracker + plotting
  main.py           # CLI runner for full ablation sweeps
  quick_test.py     # Single experiment runner
  tasks/            # Task implementations
    base.py         # BaseTask ABC + dataset utilities
    sorting.py      # Sequence sorting
    reversal.py     # Sequence reversal
    modular_arith.py# Modular sum (mod p)
    bitwise_add.py  # Binary addition (carry propagation)
    parity.py       # XOR parity + running XOR chain

utils/              # General utilities
  data_utils.py     # Wikitext-2 data loading
  training.py       # WSD learning rate schedule
```

## Setup

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

## Quick Test

```bash
python -m ablation_suite.quick_test --task sorting --model baseline --epochs 5
```

## Full Ablation

```bash
python -m ablation_suite.main --all --epochs 10
```
