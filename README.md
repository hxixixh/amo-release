# README.md

## Overview

This code provides the implementation of the **Overshooting Sampler** as part of the supplementary material for our paper. The file `overshooting_sampler.py` contains the implementation of the overshooting algorithm, designed as a scheduler subclass within the [Diffusers library](https://github.com/huggingface/diffusers). This custom scheduler can be used to perform overshooting sampling, offering improved performance in text-to-image generation tasks.

## Installation and Usage

### Prerequisites

Before using the overshooting sampler, ensure you have the [Diffusers library](https://github.com/huggingface/diffusers) installed. You can install it via pip:

```bash
pip install diffusers
```

Additionally, ensure that your Python environment includes the necessary dependencies for Diffusers, such as PyTorch.

### Adding the Overshooting Sampler

1. **Copy the File**  
   Place the provided `overshooting_sampler.py` file into the `diffusers/schedulers` directory within your local Diffusers installation.

   For example:
   ```bash
   cp overshooting_sampler.py /path/to/diffusers/schedulers/
   ```

2. **Modify Your Code**  
   To use the overshooting sampler in your project, replace the default scheduler (e.g., `EulerDiscreteScheduler`) with the `StochasticRFOvershotDiscreteScheduler`. Below is an example:

   ```python
   from diffusers.schedulers import StochasticRFOvershotDiscreteScheduler

   scheduler = StochasticRFOvershotDiscreteScheduler(...)
   ```

3. **Run Your Model**  
   Replace the default scheduler in your pipeline configuration with the overshooting sampler and run your model as usual. This will apply the overshooting sampling algorithm.

### Implementation Details

- The `overshooting_sampler.py` file extends the existing scheduler framework in Diffusers.
- It introduces a soft overshooting mechanism to improve text rendering and image quality.
- Users can configure overshooting-specific parameters, such as the overshooting strength (`c`), in the sampler initialization.



```bash
conda create -n amo python=3.12
# install pytorch that match your own cuda version, We used 2.5.1 
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install diffusers==0.30.1
pip install transformers==4.44.2 
pip install accelerate==1.4.0
pip install sentencepiece==0.2.0
pip install protobuf==5.29.3


```

```bash
python run.py
```