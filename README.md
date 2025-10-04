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


---

## Key Differences from BackRazor

LoRA Support: This fork adds support for LoRA (Low-Rank Adaptation), enabling efficient fine-tuning with fewer parameters and reduced memory usage.

## Logging:

Console logs (.txt), runtime, and energy usage (.csv) are stored in the logs/ folder.

TensorBoard event files (.tfevents) are saved under logs/(model_name)/.

## Running Different Modes

Training modes like ftfull, bitfit, backrazor, and lora are configured via scripts in the cmds/ folder.

For LoRA training, add the following flags:
--train_lora_only \
--lora_rank 4 \
--lora_alpha 8 \
--lora_dropout 0.1

For more details on other training configurations, see the BackRazor README

## Citation
@inproceedings{jiang2022back,
  title        = {Back Razor: Memory-Efficient Transfer Learning by Self-Sparsified Backpropagation},
  author       = {Jiang, Ziyu and Chen, Xuxi and Huang, Xueqin and Du, Xianzhi and Zhou, Denny and Wang, Zhangyang},
  booktitle    = {Advances in Neural Information Processing Systems 35 (NeurIPS 2022)},
  year         = {2022},
}

