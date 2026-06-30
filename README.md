# CartoonDiff

Code for the paper **"Temporally-Coherent Diffusion for Cartoon Inbetweening: Noise Interpolation and Dual-Frame Matching"**, submitted to *The Visual Computer*.

## Framework

![Model Framework](./assets/fig.png)

Overall Framework of Our Model. The starting and ending frames are first processed by an encoder and concatenated with interpolated initialization noise to form the input sequence, which is then fed into the denoising network. Throughout the denoising process, information from the starting and ending frames is embedded into the network as conditional guidance via the Dual-Frame Matching module. Finally, the denoised features are decoded to produce the generated video frames.

## Code Version

This repository corresponds to the version used in the submitted manuscript.

**Git Commit:** `f97b3ccf3f7e7b4dd1c08cd57e42048232179708`

## License

This project is released under the **MIT License**. See the `LICENSE` file for details.

## Requirements

- Python == 3.10.0
- PyTorch >= 2.0.1
- CUDA >= 11.7

## Installation

1. Clone the repository.

2. Create a virtual environment.

```bash
conda create -n CartoonDiff python=3.10.0
conda activate CartoonDiff
```

3. Install the required packages.

```bash
pip3 install torch torchvision
pip install -r requirements.txt
```

## Dataset

Download the dataset from Baidu Netdisk:

Link: https://pan.baidu.com/s/1nPZRy6RLzTnN43guE5eRZw?pwd=9k8c

A permanent archive of the project resources is also available on Zenodo:

https://zenodo.org/records/19048937

## Model Weights

Download the pre-trained model weights from Baidu Netdisk:

Link: https://pan.baidu.com/s/1Nt-gz69iRv7S4rm8HrVfvg?pwd=f8s6

## Training

```bash
python train.py \
  --base ../configs/training_512_v1.0/train_config.yaml \
  --train \
  --name train \
  --logdir ../model_savepath \
  --devices 1 \
  lightning.trainer.num_nodes=1
```

## Inference

```bash
python inference.py \
  --seed 123 \
  --ckpt_path checkpoints/model.ckpt \
  --config configs/inference_512_v1.0.yaml \
  --savedir results/ \
  --n_samples 1 \
  --bs 1 \
  --height 320 \
  --width 512 \
  --unconditional_guidance_scale 7.5 \
  --ddim_steps 50 \
  --ddim_eta 1.0 \
  --prompt_dir test_data/onelast/ \
  --text_input \
  --video_length 16 \
  --frame_stride 10 \
  --timestep_spacing uniform_trailing \
  --guidance_rescale 0.7 \
  --perframe_ae \
  --interp
```

| Argument | Description |
|----------|-------------|
| `--seed` | Random seed for reproducibility (fixed to **123**) |
| `--ckpt_path` | Path to the pretrained checkpoint |
| `--config` | Inference configuration file |
| `--savedir` | Output directory |
| `--height` / `--width` | Input resolution (320 × 512) |
| `--video_length` | Number of generated frames |
| `--frame_stride` | Frame stride |
| `--unconditional_guidance_scale` | Classifier-free guidance scale |
| `--ddim_steps` | Number of DDIM sampling steps |
| `--perframe_ae` | Enable per-frame autoencoder |
| `--text_input` | Enable text-conditioned generation |

## Reproducibility

The repository includes:

- Training and inference scripts
- Configuration files
- Environment specification (`requirements.txt`)
- Dataset split files
- Evaluation scripts
- Random seed used in the paper (`123`)
- Sample outputs for qualitative comparison

The quantitative evaluation metrics (e.g., FVD, SSIM, PSNR, LPIPS, and CLIP-based metrics) are computed using the publicly available evaluation toolkit:

https://github.com/JunyaoHu/common_metrics_on_video_quality

## Citation

If you find this repository useful, please consider citing our paper.
