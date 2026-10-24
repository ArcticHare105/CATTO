# CATTO

CATTO adds inference-time condition refinement for image restoration. At selected key steps, it branches the current latent into several candidates, varies the image condition in a DCT gain space, and refines the input condition using consistency and reward scores. It uses pretrained models and does not require additional training.

The main entry point is [`scripts/run_catto.sh`](scripts/run_catto.sh). Select a configuration file from [`configs/`](configs/) based on the desired number of optimization steps, e.g., [`setting_20s.json`](configs/setting_20s.json) for 20 steps. Model and data paths are supplied when launching the script.

## 1. Environment

Clone the repository:

```bash
git clone https://github.com/ArcticHare105/CATTO.git
cd CATTO
```

The environment requirements are list as below:

```bash
conda create -n catto python=3.8 -y
conda activate catto

# CUDA 12.1 wheels; use a compatible NVIDIA driver.
python -m pip install torch==2.1.0 torchvision==0.16.0 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install \
  xformers==0.0.22.post7 diffusers==0.21.0 \
  transformers==4.37.2 accelerate==0.25.0 huggingface-hub==0.25.0 \
  timm==0.9.12 pyiqa==0.1.14.1 loralib==0.1.2 fairscale==0.4.13 \
  'numpy<2' scipy Pillow opencv-python-headless einops
```

## 2. Models and input images

Download the following pretrained weights separately:

| Model | Source | Default location |
| --- | --- | --- |
| Stable Diffusion 2 Base | [Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-2-base) | `preset/models/stable-diffusion-2-base/` |
| SeeSR and DAPE | [SeeSR model download instructions](https://github.com/cswry/SeeSR#-quick-inference) | `preset/models/seesr/` and `preset/models/DAPE.pth` |
| RAM Swin-L | [RAM checkpoint](https://huggingface.co/spaces/xinyu1205/recognize-anything/blob/main/ram_swin_large_14m.pth) | `preset/models/ram_swin_large_14m.pth` |

Arrange the files as follows. Keep the original Diffusers model files and subdirectories inside each model directory.

```text
preset/
├── models/
│   ├── stable-diffusion-2-base/
│   │   ├── scheduler/
│   │   ├── tokenizer/
│   │   ├── text_encoder/
│   │   ├── vae/
│   │   └── feature_extractor/
│   ├── seesr/
│   │   ├── unet/
│   │   └── controlnet/
│   ├── ram_swin_large_14m.pth
│   └── DAPE.pth
└── datasets/
    └── input/
        ├── image_001.png
        └── image_002.png
```

`INPUT` can be a single image or a directory. A directory is read non-recursively; put only input images in it. On the first run, RAM's `bert-base-uncased` tokenizer and PyIQA's CLIPIQA assets may also need to download. Cache these resources beforehand for offline inference.

## 3. Run inference

With the default directory layout:

```bash
bash scripts/run_catto.sh
```

To set the GPU and paths:

```bash
GPU=0 \
MODEL_ROOT=/path/to/models \
INPUT=/path/to/input_images \
OUTPUT=outputs/catto \
bash scripts/run_catto.sh
```

The script loads `configs/setting.json` automatically. To use another configuration, set `CONFIG`:

```bash
CONFIG=configs/my_setting.json bash scripts/run_catto.sh
```

Environment variables accepted by the script:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_ROOT` | `preset/models` | Base directory for the four model paths. |
| `SD_MODEL` | `$MODEL_ROOT/stable-diffusion-2-base` | Override the SD model directory. |
| `SEESR_MODEL` | `$MODEL_ROOT/seesr` | Override the SeeSR model directory. |
| `RAM_MODEL` | `$MODEL_ROOT/ram_swin_large_14m.pth` | Override the RAM checkpoint. |
| `DAPE_MODEL` | `$MODEL_ROOT/DAPE.pth` | Override the DAPE checkpoint. |
| `INPUT` | `preset/datasets/input` | Input image or directory. |
| `OUTPUT` | `outputs/catto` | Output directory. |
| `CONFIG` | `configs/setting.json` | CATTO JSON configuration. |

## :smiley: Citation

Please cite us if our work is useful for your research.

```
@article{2024s3diff,
  author    = {Aiping Zhang, Jiangang Wang, Shangquan Sun, Yuning Cui, Fan Li, Renjing Pei, Wenqi Ren, Xiaochun Cao},
  title     = {Beyond Single-Shot Conditioning: Test-Time Condition Refinement for Diffusion-Based Image Restoration},
  journal   = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## :notebook: License

This project is released under the [Apache 2.0 license](LICENSE).


## :envelope: Contact

If you have any questions, please feel free to contact zhangaip7@mail2.sysu.edu.cn.
