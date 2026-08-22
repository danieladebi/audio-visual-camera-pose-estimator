"""
Self-supervised mono-to-binaural ("M2B") pretraining, inspired by SLfM
(Self-Supervised Localization from Motion, https://github.com/IFICL/SLfM).

Why this script exists
-----------------------
The shipped checkpoint
`checkpoints/_egoexo4d-vision_slfm_embed_and_doa_360_1000ms-512_/checkpoint-best.pth`
expects a 1024-dim "SLfM" audio embedding per frame (concatenated with a 360-dim
DOA vector -> `audio_embed_proj.0.weight` shape [1024, 1384]). The original cache
that produced those 1024-dim embeddings (`embeddings_slfm_egoexo4d/`) no longer
exists and the exact generation recipe is lost. The repo owner authorized
reconstructing it via a FRESH, from-scratch self-supervised training run (not by
loading any pretrained SLfM checkpoint), as long as the final embedding is a real
1024-dim float32 vector coming from an actual training process on real audio.

Architecture
------------
We reuse `CondAudioEncoder` and `AudioCondUNet` *verbatim* from
`reloc3r/reloc3r_relpose.py` (this repo already contains clean, dependency-free
copies of these exact SLfM model classes -- no need to import from the SLfM repo
itself, which has import-time side effects/heavy deps like habitat_sim pulled in
through its own `config`/`utils` packages).

`AudioCondUNet` is hard-coded at construction time for the *tri-modal* case
(no_vision=False, no_cond_audio=False -> cond_feat_dim = 512 (vision) + 512
(audio) = 1024, bottleneck width 512+1024=1536). We only want the *audio-only*
variant (no_vision=True) described in the task, so `AudioOnlyCondUNet` below
subclasses it and only rebuilds the one layer whose shape depends on
`cond_feat_dim` (`audionet_upconvlayer1`), giving a clean audio-only bottleneck
of 512 (own conv5 features) + 512 (CondAudioEncoder conditioning) = 1024
channels at an 8x8 spatial resolution (256x256 spectrogram input, 5 stride-2
convs: 256->128->64->32->16->8).

Self-supervised objective (mono -> binaural, "M2B" pretext task)
------------------------------------------------------------------
For every 1-second 7-channel mic-array clip in
`/vision/vision_data_2/egoexo4d_audio/audio_clips_1s/{take}/{take}_{frame:06d}.wav`:
  - mono   = mean of all 7 channels                      (this is the "input")
  - binaural = channels [5, 6] (0-indexed), following the precedent in
    `extract_binaural_audio.py` / `egoexo4d_audio_doa.py`     (this is the "target")

We predict the complex spectrogram of the binaural *difference* signal
(L - R) from a fixed-size (256x256) real/imag spectrogram of the mono signal,
using AudioCondUNet's complex-mask mechanism (exactly the "mono2binaural"
branch already implemented in `AudioCondUNet.forward`). The conditioning
feature fed to `CondAudioEncoder` is the SAME mono clip, duplicated to two
"channels" (mono, mono) purely to satisfy `CondAudioEncoder`'s hard-coded
4-channel (2 audio channels x [mag,phase]) conv1 -- i.e. this is a genuinely
*self-conditioned* mono -> binaural task: at no point does the network see the
binaural target as an input, so there is no shortcut/leakage. This mirrors how
the embedding will later be extracted at inference/caching time (self-conditioned
on a single frame's own audio, see `generate_embeddings_slfm_egoexo4d.py`).

Loss: L1 between the predicted and ground-truth (L-R) difference spectrograms
(real/imag channels stacked) -- a direct analogue of SLfMNet.calc_loss's
'L1' spec loss, just computed with our own dataset/preprocessing since SLfM's
own data pipeline (data/slfm_base_loader.py, main.py) is tightly coupled to a
synthetic AI-Habitat/HM3D dataset with known source geometry that does not
apply to our real in-the-wild recordings.

Deviations from the literal task description (and why)
--------------------------------------------------------
- STFT config: we do NOT reuse SLfM's `n_fft=512, hop_length=160, win_length=400`
  verbatim, because that hop_length was tuned for their 2.55s clips. Instead we
  reuse the *shape trick* already used inside `CondAudioEncoder.wave2spec`
  (n_fft=512, win_length=400, but hop_length computed dynamically so the output
  has exactly 256 time frames) so that our own 1s clips also produce a clean
  256x256 spectrogram, giving an exact 8x8 bottleneck after 5 stride-2 downconvs
  -- this is what the task description flags as "the natural candidate" shape.
  Internal consistency (all our own code, matching shapes end-to-end) mattered
  more here than matching SLfM's literal numbers, since we are training fresh.
- Audio is loaded at its native 48kHz (verified via `sf.info`) and resampled to
  16kHz (SLfM's convention, also reused by `egoexo4d.py`'s own `use_slfm` branch)
  before any spectrogram computation.
"""
import argparse
import glob
import os
import random
import sys
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reloc3r.reloc3r_relpose import CondAudioEncoder, AudioCondUNet, unet_upconv  # noqa: E402

# NOTE on the audio source directory: the task brief originally pointed at
# `audio_clips_1s/{take}/{take}_{frame_id:06d}.wav`, but that directory uses a
# DIFFERENT frame-numbering grid (0, 30, 60, ...) than the one actually used by
# `reloc3r/datasets/egoexo4d.py`'s frame pairs (source_frame_id/target_frame_id,
# e.g. 15, 45, 75, ...) and by the DOA cache under
# `camera_pose_audio_data/doa/{take}/doa_{frame_id:06d}_duration_*ms.npy`
# (constant off-by-15 offset, verified empirically for cmu_bike01_2/4,
# cmu_bike02_2). Embeddings keyed by the `audio_clips_1s` frame grid would
# never be looked up by the loader. We instead use
# `camera_pose_audio_data/audio/{take}/sound/{frame_id:06d}_duration_1000ms.wav`,
# which has the SAME format (48kHz, 7ch, 1s = 48000 samples) but is keyed by
# the frame ids that the frame-pairs pickle / DOA cache actually use.
AUDIO_CLIPS_ROOT = "/vision/vision_data_2/egoexo4d_audio/camera_pose_audio_data/audio"
SOUND_DURATION_MS = 1000
NATIVE_SR = 48000
TARGET_SR = 16000
N_FFT = 512
WIN_LENGTH = 400
SPEC_FRAMES = 256  # -> 8x8 bottleneck after 5 stride-2 downconvs (256/2**5=8)
BINAURAL_CHANNELS = (5, 6)  # per extract_binaural_audio.py / egoexo4d_audio_doa.py precedent


def wave2spec_ri(wave, n_fft=N_FFT, win_length=WIN_LENGTH, frames=SPEC_FRAMES):
    """(N, C, L) real waveform -> (N, C*2, F, T) real/imag spectrogram, F=256, T=frames.

    Same 'fixed output size via dynamic hop_length' trick used in
    CondAudioEncoder.wave2spec, generalized to return real/imag (needed by
    AudioCondUNet's complex-mask math) instead of mag/phase.
    """
    N, C, L = wave.shape
    wave_flat = wave.reshape(N * C, L)
    hop_length = max(1, L // (frames - 1))
    window = torch.hann_window(win_length, device=wave.device, dtype=wave.dtype)
    spec = torch.stft(wave_flat, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                       window=window, return_complex=True)  # (N*C, F, T')
    spec = spec.reshape(N, C, spec.shape[-2], spec.shape[-1])  # (N, C, F, T')
    T = spec.shape[-1]
    if T < frames:
        spec = F.pad(spec, (0, frames - T))
    else:
        spec = spec[..., :frames]
    spec_ri = torch.view_as_real(spec)  # (N, C, F, frames, 2)
    spec_ri = spec_ri.permute(0, 1, 4, 2, 3).contiguous()  # (N, C, 2, F, frames)
    spec_ri = spec_ri.view(N, C * 2, spec_ri.shape[-2], frames)  # (N, C*2, F, frames)
    spec_ri = spec_ri[:, :, :-1, :]  # drop last freq bin: 257 -> 256
    return spec_ri


class AudioOnlyCondUNet(AudioCondUNet):
    """AudioCondUNet configured for the audio-only (no_vision=True) mono2binaural
    pretext task. See module docstring for why we can't just pass a flag to
    the base class's __init__."""

    def __init__(self, ngf=64, audio_feature_dim=512):
        super().__init__(ngf=ngf)
        self.no_vision = True
        self.no_cond_audio = False
        cond_feat_dim = audio_feature_dim  # audio-only: no visual_feature_dim term
        self.audio_visual_feat_dim = ngf * 8 + cond_feat_dim  # 512 + 512 = 1024
        # only this layer's input channel count depends on cond_feat_dim
        self.audionet_upconvlayer1 = unet_upconv(self.audio_visual_feat_dim, ngf * 8)

    def encode_bottleneck(self, input_audio):
        """Return the (N, 512, 8, 8) conv5 bottleneck feature (pre cond-concat)."""
        x = self.audionet_convlayer1(input_audio)
        x = self.audionet_convlayer2(x)
        x = self.audionet_convlayer3(x)
        x = self.audionet_convlayer4(x)
        x = self.audionet_convlayer5(x)
        return x

    def forward(self, input_audio, cond_feats):
        """Same computation as `AudioCondUNet.forward` (mono2binaural complex-mask
        branch), but with a size-safe trim instead of the parent's hard-coded
        `[..., :-1]` time-axis slice.

        The parent class's `[..., :-1]` assumes `mask_prediction`'s time
        dimension is exactly one greater than what's needed to line up with
        `input_audio[..., :-1]` -- true for SLfM's original (non power-of-2)
        STFT frame counts, but NOT generally true here: our `wave2spec_ri`
        deliberately produces a clean power-of-2 (256x256) spectrogram (see
        module docstring) so the 5 stride-2 downconvs / upconvs reconstruct the
        time axis exactly (no parity loss), leaving `mask_prediction` the SAME
        width as `input_audio`, not one narrower. We therefore trim both
        tensors to their common time length instead of assuming a fixed offset.
        This is the only functional change relative to the verbatim
        `AudioCondUNet` implementation in `reloc3r/reloc3r_relpose.py`.
        """
        audio_conv1feature = self.audionet_convlayer1(input_audio)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)

        cond_feats_b = cond_feats.view(cond_feats.size(0), -1, 1, 1).repeat(
            1, 1, audio_conv5feature.shape[-2], audio_conv5feature.shape[-1])
        audioVisual_feature = torch.cat((cond_feats_b, audio_conv5feature), dim=1)

        audio_upconv1feature = self.audionet_upconvlayer1(audioVisual_feature)
        audio_upconv2feature = self.audionet_upconvlayer2(torch.cat((audio_upconv1feature, audio_conv4feature), dim=1))
        audio_upconv3feature = self.audionet_upconvlayer3(torch.cat((audio_upconv2feature, audio_conv3feature), dim=1))
        audio_upconv4feature = self.audionet_upconvlayer4(torch.cat((audio_upconv3feature, audio_conv2feature), dim=1))
        prediction = self.audionet_upconvlayer5(torch.cat((audio_upconv4feature, audio_conv1feature), dim=1))

        mask_prediction = torch.sigmoid(prediction) * 2 - 1  # (N, 2, F, T_pred)
        T = min(input_audio.shape[-1] - 1, mask_prediction.shape[-1])
        real_in, imag_in = input_audio[:, 0, :, :T], input_audio[:, 1, :, :T]
        real_mask, imag_mask = mask_prediction[:, 0, :, :T], mask_prediction[:, 1, :, :T]
        spec_diff_real = real_in * real_mask - imag_in * imag_mask
        spec_diff_img = real_in * imag_mask + imag_in * real_mask
        return torch.cat((spec_diff_real.unsqueeze(1), spec_diff_img.unsqueeze(1)), dim=1)


class SLfMM2BModel(nn.Module):
    """Audio-only self-supervised mono->binaural model: CondAudioEncoder (for the
    self-conditioning feature) + AudioOnlyCondUNet (for the mono->binaural
    complex-mask prediction)."""

    def __init__(self):
        super().__init__()
        self.audio_net = CondAudioEncoder(audio_backbone='resnet18', audio_feature_dim=512,
                                           n_fft=N_FFT, win_length=WIN_LENGTH)
        self.audio_net.cond_clip_length = 1.0  # our clips are 1s @ 16kHz
        self.audio_net.samp_sr = TARGET_SR
        self.generative_net = AudioOnlyCondUNet(audio_feature_dim=512)

    def forward(self, mono_cond, mono_spec):
        """
        mono_cond: (N, 2, L) -- mono waveform duplicated across 2 "channels" so it
                   satisfies CondAudioEncoder's hard-coded 4-in-channel conv1
        mono_spec: (N, 2, 256, 256) -- real/imag spectrogram of the mono waveform
        returns: pred_diff_spec (N, 2, 256, 255), embedding (N, 1024)
        """
        cond_feats = self.audio_net(mono_cond)  # (N, 512)
        pred_diff_spec = self.generative_net(mono_spec, cond_feats)  # (N, 2, 256, 255)

        bottleneck = self.generative_net.encode_bottleneck(mono_spec)  # (N, 512, 8, 8)
        bottleneck_pooled = F.adaptive_avg_pool2d(bottleneck, 1).flatten(1)  # (N, 512)
        embedding = torch.cat([bottleneck_pooled, cond_feats], dim=1)  # (N, 1024)
        return pred_diff_spec, embedding


def list_take_clips(take, clips_root=AUDIO_CLIPS_ROOT, duration_ms=SOUND_DURATION_MS):
    """(take, frame_id, wav_path) tuples for a take, keyed by the frame-id grid
    actually used by the frame-pairs pickle / DOA cache (see note above)."""
    pattern = os.path.join(clips_root, take, "sound", f"*_duration_{duration_ms}ms.wav")
    items = []
    for wav_path in sorted(glob.glob(pattern)):
        fname = os.path.basename(wav_path)
        frame_id = int(fname.split("_duration_")[0])
        items.append((take, frame_id, wav_path))
    return items


class M2BClipDataset(Dataset):
    """Enumerates all 1s mic-array clips for a list of takes."""

    def __init__(self, takes, clips_root=AUDIO_CLIPS_ROOT):
        self.items = []  # (take_name, frame_id, wav_path)
        for take in takes:
            self.items.extend(list_take_clips(take, clips_root))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        take, frame_id, wav_path = self.items[idx]
        wave, sr = sf.read(wav_path, dtype='float32')  # (L, 7)
        wave = torch.from_numpy(wave.T)  # (7, L)
        if sr != TARGET_SR:
            wave = torchaudio.functional.resample(wave, sr, TARGET_SR)
        mono = wave.mean(dim=0)  # (L,)
        binaural = wave[list(BINAURAL_CHANNELS)]  # (2, L)
        return {
            'take': take,
            'frame_id': frame_id,
            'mono': mono,
            'binaural': binaural,
        }


def collate(batch):
    lens = [b['mono'].shape[-1] for b in batch]
    L = min(lens)  # clips should all be ~1s but guard against tiny length mismatches
    mono = torch.stack([b['mono'][:L] for b in batch])  # (N, L)
    binaural = torch.stack([b['binaural'][:, :L] for b in batch])  # (N, 2, L)
    takes = [b['take'] for b in batch]
    frame_ids = [b['frame_id'] for b in batch]
    return mono, binaural, takes, frame_ids


def compute_loss(model, mono, binaural, device):
    mono = mono.to(device)
    binaural = binaural.to(device)
    N = mono.shape[0]

    mono_cond = torch.stack([mono, mono], dim=1)  # (N, 2, L) fake-stereo for CondAudioEncoder
    mono_spec = wave2spec_ri(mono.unsqueeze(1))  # (N, 2, 256, 256)

    pred_diff_spec, embedding = model(mono_cond, mono_spec)

    diff_wave = (binaural[:, 0] - binaural[:, 1]).unsqueeze(1)  # (N, 1, L)
    target_diff_spec = wave2spec_ri(diff_wave)  # (N, 2, 256, 256)
    T = pred_diff_spec.shape[-1]
    target_diff_spec = target_diff_spec[..., :T]

    loss = F.l1_loss(pred_diff_spec, target_diff_spec)
    return loss, embedding


def main():
    parser = argparse.ArgumentParser(description="Validation-scale self-supervised SLfM-style mono2binaural training")
    parser.add_argument('--takes', type=str, required=True,
                         help='comma-separated list of take names to train on')
    parser.add_argument('--val_frac', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='checkpoints/slfm_m2b_egoexo4d')
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    takes = [t.strip() for t in args.takes.split(',') if t.strip()]
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = M2BClipDataset(takes)
    print(f"Total clips across {len(takes)} takes: {len(dataset)}")
    n_val = max(1, int(len(dataset) * args.val_frac))
    idx = list(range(len(dataset)))
    random.shuffle(idx)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)
    print(f"Train clips: {len(train_set)}, Val clips: {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, collate_fn=collate, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate, drop_last=False)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = SLfMM2BModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float('inf')
    history = []
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        train_losses = []
        for mono, binaural, _, _ in train_loader:
            optimizer.zero_grad()
            loss, _ = compute_loss(model, mono, binaural, device)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = float(np.mean(train_losses)) if train_losses else float('nan')

        model.eval()
        val_losses = []
        with torch.no_grad():
            for mono, binaural, _, _ in val_loader:
                loss, _ = compute_loss(model, mono, binaural, device)
                val_losses.append(loss.item())
        val_loss = float(np.mean(val_losses)) if val_losses else float('nan')
        dt = time.time() - t0

        print(f"epoch {epoch:03d}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
              f"time={dt:.2f}s  ({dt / max(1, len(train_set)):.4f}s/clip)")
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'time': dt})

        ckpt = {'model': model.state_dict(), 'epoch': epoch, 'args': vars(args), 'history': history}
        torch.save(ckpt, os.path.join(args.output_dir, 'checkpoint-last.pth'))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(args.output_dir, 'checkpoint-best.pth'))

    print(f"Done. Best val loss: {best_val:.6f}")


if __name__ == '__main__':
    main()
