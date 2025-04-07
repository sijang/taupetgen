
# TauPETGen: Text-Conditional Tau PET Image Synthesis Based on Latent Diffusion Models

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2306.11984)

This repository contains code for generating tau PET images using a custom Stable Diffusion model.


### Installation


```python
conda env create -f environment.yml
```

### How to do inference

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
fileName = "datasets/mr_example1.png"

# Cell 3: Run inference
# Generate images for later stage
infer_later_mmse(pipe, fileName, image_guidance_scale, guidance_scale, num_inference_steps, generator)

fileName = "datasets/tau_later_example1.png"

# Cell 4: Run inference
# Generate images for later stage
infer_mr(pipe, fileName, image_guidance_scale, guidance_scale, num_inference_steps, generator)

```


### How to train
Please complete the config for accelerate
```python
accelerate config
```

```python
accelerate launch --mixed_precision="fp16"  train_text_to_image-instruct.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --train_data_dir="datasets" \
  --use_ema \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=1000\
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="train_save"
```
