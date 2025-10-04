# BackRazor LoRA

This repository builds on [BackRazor (NeurIPS 2022)](https://github.com/VITA-Group/BackRazor_Neurips22) with additional modifications and files for LoRA-based experiments.

---

## Installation

Follow the **original BackRazor installation instructions** from the official repo:  
[VITA-Group/BackRazor_Neurips22](https://github.com/VITA-Group/BackRazor_Neurips22)

That will set up the correct environment, dependencies, and data preparation steps.

---

## Usage with this Repo

After completing the installation steps above:

1. Clone this repository:
   ```bash
   git clone https://github.com/mirasomv/backrazor-lora.git
   cd backrazor-lora
2. Copy/replace the provided files into your BackRazor installation
3. Download [ViT-S/16](https://github.com/google-research/vision_transformer) and put it into pretrain/ folder
4. Run experiments as described in the original BackRazor instructions, now with the updated files.

This fork adds support for LoRA (Low-Rank Adaptation), allowing efficient fine-tuning with reduced parameter count and memory usage. If using the same cmd, the results will be saved as follows:
Console log (.txt) + time + Energy usage (.csv) in logs/ folder
TFEvents in logs/(model name)/ folder
