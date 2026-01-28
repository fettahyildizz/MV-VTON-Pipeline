# MV-VTON-Pipeline (monorepo)

This repository is a **single Git repo** that contains three research codebases that work together:

- [MV-VTON](MV-VTON/) — diffusion-based virtual try-on
- [pytorch-openpose](pytorch-openpose/) — body pose estimation (keypoints + rendered skeleton)
- [Self-Correction-Human-Parsing](Self-Correction-Human-Parsing/) — human parsing/segmentation (SCHP)

The intent is: **OpenPose + SCHP produce conditioning inputs** (pose + parsing), then **MV-VTON runs inference** with environment support for the latest GPUs with Blackwell architecture (RTX50s) and Ubuntu 24.04. 

Since the original code is written for old CUDA and Torch versions, I also updated the code that it now may work on CUDA 13.0 and on the latest Torch versions and for Ubuntu 24.04, supports Python 3.9

## Installation

1. Clone the repository

```shell
git clone https://github.com/fettahyildizz/MV-VTON-Pipeline.git
cd MV-VTON-Pipeline
```

2. Install Python dependencies

```shell
conda env create -f requirement.yaml -n mv-vton-pip
conda activate mv-vton-pip
```

3. Download all models required by checking Readmes of sub-repos. 
- You should be downloading one model for pytorch-openpose named `body_pose_model.pth`
- You should be downloading one model for schp named `atr-schp-201908301523-atr.pth`
- You should be downloading two models for mv-vton named `mvg.ckpt` and `vgg19_conv.pth`.

## Running the pipeline (examples)

### MV-VTON inference

#### Single-pair mode

This mode takes one person image and one cloth image and automatically runs SCHP + OpenPose internally.

Example:

```bash
python MV-VTON/inference.py \
    --config MV-VTON/configs/viton512.yaml \
    --ckpt MV-VTON/checkpoints/mvg.ckpt \
    --person-image MV-VTON/images/person/person1.png \
    --cloth-image MV-VTON/images/cloth/tshirt1.png \
    --openpose-model-dir pytorch-openpose/model \
    --schp-ckpt Self-Correction-Human-Parsing/models/atr-schp-201908301523-atr.pth \
    --single-folder single_1 \
    --ddim_steps 30 \
```

Notes:

- If you have a cloth mask, pass `--cloth-mask /path/to/mask.png`.
- Outputs will be written under `--outdir`. Default outputs path is outputs/infer path. It will be automatically created after an inference pass.
- If you run out of CUDA memory, pass `--use_fp16` flag for float16 precison to reduce memory consumption.
- You may want to increase or decrease ddim_steps, limited by gpu consumption. It may affect the quality of the outputs.

## TODO:
* I may add inference code.

## License

This project is licensed under the MIT License.

Fettah YILDIZ