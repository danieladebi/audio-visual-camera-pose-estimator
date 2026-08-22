"""
Generate the `embeddings_slfm_egoexo4d/` cache expected by
`reloc3r/datasets/egoexo4d.py`'s `use_slfm_and_doa` branch, using a checkpoint
trained by `train_slfm_m2b_egoexo4d.py`.

Loader contract (reloc3r/datasets/egoexo4d.py ~line 517-522):
    label = f"{take_name}_{frame_id:06d}"
    embedding1 = torch.load(f"embeddings_slfm_egoexo4d/{take}/embedding1_{label}.pt").astype(np.float32).squeeze()
    embedding2 = torch.load(f"embeddings_slfm_egoexo4d/{take}/embedding2_{label}.pt").astype(np.float32).squeeze()

Note `.astype(...)` is a *numpy* method (torch.Tensor has no `.astype` in the
torch version used by this repo's `reloc3r` conda env, verified empirically) --
so the cache must store numpy arrays (saved via `torch.save`, which pickles
arbitrary picklable objects, not just tensors), not torch Tensors. We follow
that convention here.

"embedding1" vs "embedding2": in the loader these are simply the audio
embedding for the *source* frame and for the *target* frame of a training
pair, respectively -- i.e. per-frame embeddings referenced under a role-specific
filename, not two different embeddings of the same frame. Since at
cache-generation time we don't know in advance which frames will play which
role for which pairs (and a given frame can be a "source" in one pair and a
"target" in another), we save the SAME per-frame embedding under both
`embedding1_{label}.pt` and `embedding2_{label}.pt` for every frame. This
matches the loader's usage (`embedding1_{source_label}.pt` /
`embedding2_{target_label}.pt`, both looked up by the *same* labeling
convention `{take}_{frame_id:06d}`) and is the only assumption consistent with
having one embedding per frame. Flagging this explicitly per the task
instructions in case this assumption is wrong.

Self-conditioning at inference time: unlike training (which never needs a
second, independent audio source since a single frame's own mono/binaural
split already suffices), *extraction* only has one frame's own audio
available (no separate "query" frame), so we self-condition: the same frame's
own mono clip is used both as the conditioning input to CondAudioEncoder and
as the thing being encoded (bottleneck) -- see `SLfMM2BModel.forward` reuse
below.
"""
import argparse
import glob
import os
import sys

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets_preprocess.train_slfm_m2b_egoexo4d import (  # noqa: E402
    SLfMM2BModel, wave2spec_ri, AUDIO_CLIPS_ROOT, TARGET_SR, list_take_clips,
)

OUTPUT_ROOT = "embeddings_slfm_egoexo4d"


def load_model(checkpoint_path, device):
    model = SLfMM2BModel().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model


@torch.no_grad()
def embed_clip(model, wav_path, device):
    wave, sr = sf.read(wav_path, dtype='float32')  # (L, 7)
    wave = torch.from_numpy(wave.T)  # (7, L)
    if sr != TARGET_SR:
        wave = torchaudio.functional.resample(wave, sr, TARGET_SR)
    mono = wave.mean(dim=0).unsqueeze(0).to(device)  # (1, L)

    mono_cond = torch.stack([mono, mono], dim=1)  # (1, 2, L) self-conditioned
    mono_spec = wave2spec_ri(mono.unsqueeze(1))  # (1, 2, 256, 256)

    _, embedding = model(mono_cond, mono_spec)  # (1, 1024)
    return embedding.squeeze(0).cpu().numpy().astype(np.float32)  # (1024,) numpy float32


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings_slfm_egoexo4d/ cache")
    parser.add_argument('--checkpoint', type=str,
                         default='checkpoints/slfm_m2b_egoexo4d/checkpoint-best.pth')
    parser.add_argument('--takes', type=str, required=True,
                         help='comma-separated list of take names to extract embeddings for')
    parser.add_argument('--clips_root', type=str, default=AUDIO_CLIPS_ROOT)
    parser.add_argument('--output_root', type=str, default=OUTPUT_ROOT)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = load_model(args.checkpoint, device)

    takes = [t.strip() for t in args.takes.split(',') if t.strip()]
    total = 0
    for take in takes:
        out_dir = os.path.join(args.output_root, take)
        os.makedirs(out_dir, exist_ok=True)
        clips = list_take_clips(take, clips_root=args.clips_root)
        for _, frame_id, wav_path in clips:
            label = f"{take}_{frame_id:06d}"
            emb = embed_clip(model, wav_path, device)
            torch.save(emb, os.path.join(out_dir, f"embedding1_{label}.pt"))
            torch.save(emb, os.path.join(out_dir, f"embedding2_{label}.pt"))
            total += 1
        print(f"{take}: wrote {len(clips)} frame embeddings -> {out_dir}")
    print(f"Done. Total frames embedded: {total}")


if __name__ == '__main__':
    main()
