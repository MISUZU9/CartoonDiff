# CartoonDiff
Code for the paper "Temporally-Coherent Diffusion for Cartoon Inbetweening: Noise Interpolation and Dual-Frame Matching".

## Framework

![Model Framework](./assets/fig.png)

Overall Framework of Our Model. The starting and ending frames are first processed by an encoder and concatenated with interpolated initialization noise to form the input sequence, which is then fed into the denoising network. Throughout the denoising process, information from the starting and ending frames is embedded into the network as conditional guidance via the Dual-Frame Matching module. Finally, the denoised features are decoded to produce the generated video frames.

## Requirements
- Python == 3.10.0
- PyTorch >= 2.0.1
- CUDA >= 11.7

## Installation
1. Clone the repository
2. Create a virtual environment
```bash
conda create -n CartoonDiff python=3.10.0
conda activate CartoonDiff
```
3. Install required packages
```bash
pip3 install torch torchvision
pip install -r requirements.txt
```

## Dataset
Download the dataset from Baidu Netdisk:  
Link: https://pan.baidu.com/s/1nPZRy6RLzTnN43guE5eRZw?pwd=9k8c  

## Model Weights
Download the pre-trained model weights from Baidu Netdisk:  
Link: https://pan.baidu.com/s/1Nt-gz69iRv7S4rm8HrVfvg?pwd=f8s6  

## Inference
Run the inference script with the following arguments:
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

| Argument                          | Description                                                                 |
|-----------------------------------|-----------------------------------------------------------------------------|
| `--seed`                          | Random seed for result reproducibility (fixed to 123)                        |
| `--ckpt_path`                     | Path to the pre-trained model checkpoint file                               |
| `--config`                        | Path to the inference configuration YAML file                               |
| `--savedir`                       | Directory to save generated video frames/results                            |
| `--height`/`--width`              | Resolution of the generated cartoon frames (320×512)                        |
| `--video_length`                  | Total number of frames in the generated video sequence                      |
| `--frame_stride`                  | Stride between consecutive frames in interpolation                          |
| `--unconditional_guidance_scale`  | Guidance scale for unconditional generation (7.5)                           |
| `--ddim_steps`                    | Number of DDIM sampling steps (50)                                           |
| `--perframe_ae`                   | Use per-frame autoencoder for feature processing                            |
| `--text_input`                    | Enable text prompt-based generation                                          |