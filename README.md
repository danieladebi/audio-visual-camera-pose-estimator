<h2>[ECCV 2026] Audio-Visual Camera Pose Estimation with Passive Scene Sounds and In-the-Wild Video</h2>
 
<p align="center">
<a href="https://danieladebi.com/">Daniel Adebi</a>   
 ·
<a href="https://sagnikmjr.github.io/">Sagnik Majumder</a>
 ·
<a href="https://www.cs.utexas.edu/~grauman/">Kristen Grauman</a>
</p>
<div align="center"> The University of Texas at Austin 

<h3>
<a href="https://vision.cs.utexas.edu/projects/av_camera_pose/"><strong>Project Page</strong></a>
&nbsp;|&nbsp;
<a href="https://arxiv.org/abs/2512.12165"><strong>arXiv</strong></a>
</h3>
</div>

<p align="center">
<img src="eccv_av_cpe_2026.png" width="80%">
</p>

This repository contains the code and instructions to train our model for this project.

## Overview


We extend a vision-only relative-pose regressor (built on [Reloc3r](https://github.com/ffrivera0/reloc3r) / [DUSt3R](https://github.com/naver/dust3r)) with audio cues derived from the scene's ambient sound. Our released method fuses, for each image pair from the [Ego-Exo4D](https://ego-exo4d-data.org/) dataset:


- **Vision** — the two RGB frames,
- **DOA** — direction-of-arrival spectrums of the ambient audio, and
- **Binaural embeddings** — self-supervised binaural audio features

into a single relative-pose prediction. The diagram below displays our full method architecture.

<p align="center">
<img src="full_method_diagram.png" width="90%">
</p>


## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Data: Ego-Exo4D](#data-ego-exo4d)
- [Training](#training)
- [Evaluation](#evaluation)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Installation

1. Clone the repository (with the CroCo submodule):
```bash
git clone --recursive <your-repo-url> audio-visual-camera-pose-estimator
cd audio-visual-camera-pose-estimator
# if you already cloned without --recursive:
# git submodule update --init --recursive
```

2. Create the environment:
```bash
conda create -n avpose python=3.11 cmake=3.14.0
conda activate avpose
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia  # match your CUDA
pip install -r requirements.txt
pip install -r requirements_optional.txt   # optional: HEIC image support
```

3. (Optional) Compile the CUDA kernels for RoPE positional embeddings for faster runtime:
```bash
cd croco/models/curope/
python setup.py build_ext --inplace
cd ../../../
```

4. Download the DUSt3R backbone weights used to initialize training and place them at
`checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth`
(available from the [DUSt3R](https://github.com/naver/dust3r) project). This file is required for **training** only; evaluation loads the trained checkpoint shipped under `checkpoints/` (see below).

## Data: Ego-Exo4D

> [!IMPORTANT]
> The video and audio data are **not** included in this repository. You must download
> [**Ego-Exo4D**](https://ego-exo4d-data.org/) yourself, agree to its license, and follow the
> official instructions to obtain the takes (RGB frames + multichannel audio).

After downloading, point the dataset loader at your local copy. The relevant paths live in
`reloc3r/datasets/egoexo4d.py`:

- `DATA_ROOT` — root of the processed Ego-Exo4D camera-pose / audio data (frames, `camera_poses/`, `doa/`).
- `pairs_path` — the precomputed list of frame pairs per split.

The audio **embeddings/DOA caches** (the `embeddings_*` directories) are *not* shipped — they are
large and regenerable. The dataset reads them, per take, from a directory named by the
`embedding_type` argument (e.g. `embeddings_doa_1000ms/`) relative to the repository root.
Regenerate these features from your Ego-Exo4D download before training/evaluating.

## Training

The released method (vision + SLfM embeddings + DOA, on Ego-Exo4D) is trained with:

```bash
bash scripts/train_egoexo4d_1000ms_slfm_and_doa_360.sh
```

This launches `torchrun` across 8 GPUs and is equivalent to:

```bash
torchrun --nproc_per_node=8 train.py \
    --train_dataset "EgoExo4D(split='train', use_slfm_and_doa=True, resolution=(256, 256), embedding_type='embeddings_doa_1000ms_alt', sound_size=1000, use_img=True, transform=ColorJitter)" \
    --test_dataset  "EgoExo4D(split='val',   use_slfm_and_doa=True, resolution=(256, 256), embedding_type='embeddings_doa_1000ms_alt', sound_size=1000, use_img=True, seed=777)" \
    --model "Reloc3rRelpose(img_size=256, has_audio=True, has_audio_embedding=True, use_doa_and_embed=True)" \
    --pretrained "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth" \
    --lr 1e-5 --min_lr 1e-7 --warmup_epochs 5 --epochs 100 --batch_size 32 --accum_iter 1 \
    --save_freq 10 --keep_freq 10 --eval_freq 1 \
    --freeze_encoder \
    --output_dir "checkpoints/_egoexo4d-vision_slfm_embed_and_doa_360_1000ms-512_"
```

Checkpoints are written to `--output_dir` as `checkpoint-{epoch}.pth`, plus `checkpoint-best.pth`,
`checkpoint-last.pth`, and `checkpoint-final.pth`. The `scripts/` directory contains many additional
train/eval variants (audio-only, DOA-only, vision-only, SLfM-HM3D, policy models, etc.) used for ablations.

## Evaluation

A trained checkpoint for the released method is shipped under
`checkpoints/_egoexo4d-vision_slfm_embed_and_doa_360_1000ms-512_/`. Evaluate it on Ego-Exo4D with:

```bash
bash scripts/eval_egoexo4d_slfm_embed_and_doa_360_vision.sh
```

which runs:

```bash
python eval_relpose.py \
    --model "Reloc3rRelpose(img_size=256, has_audio=True, has_audio_embedding=True, use_doa_and_embed=True)" \
    --test_dataset "EgoExo4D(split='val', use_slfm_and_doa=True, resolution=(256, 256), embedding_type='embeddings_doa_1000ms_alt', sound_size=1000, use_img=True, seed=777)" \
    --batch_size 128
```

The exact checkpoint loaded for each model configuration is selected in
`setup_reloc3r_relpose_model()` in `reloc3r/reloc3r_relpose.py` (the active branch points at the
released SLfM+DOA+vision checkpoint). To write out per-pair pose errors / per-video AUCs, uncomment
the corresponding `np.save(...)` / `json.dump(...)` lines in `eval_relpose.py`.

To turn the saved per-video AUC files into aggregate tables/plots, use `process_aucs.py` /
`process_aucs_full.py` and the plotting scripts (`graph_line_plots.py`, `plot_ambient_audio_results.py`, etc.).


## Citation

This work was accepted to the **European Conference on Computer Vision (ECCV), 2026**. If you find it
useful in your research, please cite:

```bibtex
@inproceedings{adebi2026audiovisual,
  title     = {Audio-Visual Camera Pose Estimation with Passive Scene Sounds and In-the-Wild Video},
  author    = {Adebi, Daniel and Majumder, Sagnik and Grauman, Kristen},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2512.12165}
}
```

## Acknowledgments

Built on top of [Reloc3r](https://github.com/ffrivera0/reloc3r), [DUSt3R](https://github.com/naver/dust3r),
and [CroCo](https://github.com/naver/croco). Audio features build on self-supervised localization-from-motion (SLfM) and direction-of-arrival estimation. Data from [Ego-Exo4D](https://ego-exo4d-data.org/).


