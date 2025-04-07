
# TauPETGen: Text-Conditional Tau PET Image Synthesis Based on Latent Diffusion Models

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2306.11984)

This repository contains code for generating tau PET images using a custom Stable Diffusion model.


### Installation


```python
conda env create -f environment.yml
```

### How to use

```python
# Cell 1: Imports and setup
import torch
from src.model import setup_model
from src.inference import infer_later_mmse, infer_early_mmse
import matplotlib.pyplot as plt

%matplotlib inline
plt.rcParams['figure.figsize'] = (15, 6)

# Cell 2: Model and parameters setup
# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize model and generator
pipe, generator = setup_model(device=device)

# Parameters
image_guidance_scale = 1.5
guidance_scale = 2
num_inference_steps = 10
fileName = "example/mr_example1.png"

# Cell 3: Run inference
# Generate images for later stage
infer_later_mmse(pipe, fileName, image_guidance_scale, guidance_scale, num_inference_steps, generator)
```
