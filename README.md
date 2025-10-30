# Pi0 Triton Accelerated Inference

This project accelerates Pi0 Vision-Language-Action(VLA) model inference using Triton, significantly improving model inference performance.

## Overview

This project converts the Pi0 model from JAX to PyTorch and optimizes critical operators using Triton for efficient model inference. Key features:

- **High Performance**: Custom Triton kernels optimize key computations
- **Drop-in Replacement**: Compatible API with original Pi0 inference interface
- **Easy to Use**: Get started in just a few simple steps

## Installation

### Prerequisites
- NVIDIA GPU
- CUDA Toolkit
- Python >= 3.11
- PyTorch >= 2.7.0
- Triton 

### Install from Source

1. **Clone the repository**
```bash
git clone xxx
cd xxx
```
2. **Install dependencies**
```bash
# Install the project in editable mode
pip install -e .
```


## Usage

### 1: Weight Conversion

First, convert Pi0 JAX model weights to PyTorch format. The weight conversion functionality is integrated in the example code, including:

1. Load JAX format model weights
2. Convert to PyTorch tensors according to mapping
3. Save as `.pth` files

```python
# Weight file structure
torch_weights/
  ├── triton_pi0_vlm_weights.pth    # Vision encoder and language model weights
  └── triton_pi0_action_weights.pth  # Action decoder weights
```

Refer to the `_load_weights_from_params` method in `convert_weights.py` for weight mapping logic.

### 2: Model Inference

The project provides a `Pi0Inference` class that serves as a drop-in replacement for the original Pi0 inference interface:

```python
from pi0_triton import Pi0Inference
import torch

# 1. Load weights
vlm_weights = torch.load("triton_pi0_vlm_weights.pth")
action_weights = torch.load("triton_pi0_action_weights.pth")

# Perform weight mapping and preprocessing (refer to duifen.py)
weight_mapping = create_weight_mapping(vlm_weights, action_weights)

# 2. Initialize model
model = Pi0Inference(num_views=2, params=weight_mapping)

# 3. Prepare input data
observation_images = torch.randn(2, 3, 224, 224, dtype=torch.bfloat16, device="cuda")
observation_state = torch.randn(32, dtype=torch.bfloat16, device="cuda")
diffusion_noise = torch.randn(50, 32, dtype=torch.bfloat16, device="cuda")

# 4. Run inference
result = model.infer(observation_images, observation_state, diffusion_noise)

# 5. Get results
action_output = result['action_output']  # Predicted action sequence
elapsed_time = result['elapsed']          # Inference time (milliseconds)
```

### 3: Performance Testing

Use the `test.py` script for model performance testing:

```bash
python test.py \
    --model_type triton \
    --torch_path /path/to/torch/weights 
```

**Parameters:**
- `--model_type`: Model type, supports `triton`, `torch`, `jax`
- `--torch_path`: PyTorch weights directory


## Performance Features

### Triton Optimized Kernels

The project implements optimizations for the following key operators using Triton:

1. **Vision Encoder**
   - `conv2d_embed`: Convolutional embedding layer
   - `layer_norm_QKV_matmul`: Fused LayerNorm + QKV projection
   - `matmul_n256_*`: Shape-specific optimized matrix multiplication

2. **Transformer Decoder**
   - `matmul_k8_256_n_softmax`: Multi-head attention computation
   - `apply_rope`: Rotary position encoding
   - `FFN_matmul_gate`: Fused gated FFN operator
   - `rms_matmul_gate`: Fused RMSNorm + gated FFN
   - `fused_transformer_decoder_kernel`



## Contributing

Issues and Pull Requests are welcome!



## Acknowledgements

This project is developed based on Physical Intelligence's [OpenPI](https://github.com/Physical-Intelligence/openpi) project.


## Citation
If you want, you can cite this work with: xxx
