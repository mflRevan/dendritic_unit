
# Dendritic Unit
A library for investigating and implementing Dendritic Neural Networks.

## Overview
The **Dendritic Unit** is a drop-in replacement for the MLP blocks in Transfomers. It completely removes explicit activation functions (like ReLU, SiLU, GeLU) and replaces them with a **Competition & Coactivation** mechanism inspired by biological dendrites.

### Core Mechanism
1.  **Local Competition**: Input segments are matched against learned "templates" via dot product. The results undergo an **Absolute Softmax**, creating a winner-take-all competition where only the best-matching patterns activate.
2.  **Global Coactivation**: The output intensity is modulated by the overall "agreement" of the input with the templates (mean absolute dot product), functioning like an NMDA receptor gain.

### Variants
*   **CNN-Gated**: Efficient implementation using Depthwise Conv1d to simulate dendritic branches.
*   **Template-Gated (GLU)**: High-performance Triton kernel implementation where template matching gates a linear projection.
*   **Template-FFN**: Pure dendritic projection. The templates *are* the weights. No parallel linear layer.

## Migration & Usage
This repository contains the "clean" core implementation extracted from research code.

*   `dendritic_unit.core`: Contains the layers and Triton kernels.
*   `dendritic_unit.model`: Contains a full `DendriticTransformer` implementation.

### Example
```python
from dendritic_unit.model.transformer import DendriticTransformer

model = DendriticTransformer(
    vocab_size=32000,
    seq_length=2048,
    dim=1024,
    num_heads=16,
    num_layers=12,
    use_dendritic=True,
    use_template=True,       # Use Triton kernels
    use_dendritic_ffn=True   # Use pure dendritic projection
)
```
