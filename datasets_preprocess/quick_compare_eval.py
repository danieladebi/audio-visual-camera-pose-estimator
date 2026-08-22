"""
Quick relative-pose evaluation comparing two models on a random sample of the
REAL val split ("test split"), reusing the eval metric machinery from
`reloc3r/utils/metric.py` (get_rot_err, get_transl_ang_err, error_auc) and the
model construction / weight-loading conventions from `reloc3r_relpose.py` and
`train.py`, unmodified.

Not using `eval_relpose.py` directly because its `setup_reloc3r_relpose_model`
hard-codes model selection: for any `--model` string that doesn't contain
"DOA"/"policy"/"slfm", it ALWAYS builds
`Reloc3rRelpose(img_size=256, has_audio=True, has_audio_embedding=True, use_doa_and_embed=True)`
and loads the shipped audio+DOA checkpoint regardless of the `--model` string
passed on the CLI (everything else is commented out) -- so it can't actually
produce a vision-only baseline. `eval_relpose.py` also unconditionally looks
up `ssim_info/val_frame_pairs_ssim_info.json` for every non-SLfM sample, which
is orthogonal to what we need here. We reuse its *metric* functions directly.

Two models compared:
  1. "vision_only": Reloc3rRelpose(img_size=256, has_audio=False), quickly
     fine-tuned (frozen encoder) on 3 small validation takes --
     see `finetune_vision_only_quick.py`. Not the officially released
     vision-only baseline (that would need the full train split).
  2. "mine": the SHIPPED checkpoint
     `checkpoints/_egoexo4d-vision_slfm_embed_and_doa_360_1000ms-512_/checkpoint-best.pth`
     (Reloc3rRelpose img_size=256, has_audio=True, has_audio_embedding=True,
     use_doa_and_embed=True), fed with our RECONSTRUCTED 1024-dim SLfM
     embedding (from `checkpoints/slfm_m2b_egoexo4d/checkpoint-best.pth`,
     generated on the fly for whichever val frames get sampled here) +
     REAL 360-dim DOA vectors loaded directly from
     `camera_pose_audio_data/doa/{take}/doa_{frame:06d}_duration_1000ms.npy`.
     We build the audio_embedding tensor ourselves and inject it into the
     view dicts rather than going through `EgoExo4D(use_slfm_and_doa=True)`,
     because that code path calls `.astype(np.float32)` on whatever
     `torch.load()` returns for the DOA cache -- a numpy-only method that
     crashes on the torch.Tensor-typed `embeddings_doa_1000ms_alt/` cache
     (verified empirically). Since `egoexo4d.py` is a protected file we
     can't patch that bug, so we bypass the buggy branch entirely and
     do the equivalent concatenation ourselves, using the real DOA
     .npy files directly (numpy, no bug).

Both models are evaluated on the SAME random sample of pairs from
`val_frame_pairs_list.pickle` (the actual "test"/val split, not the 3 train
takes used for fitting either the M2B extractor or the vision-only quick
fine-tune -- disjoint by construction, since EgoExo4D's split is take-level).
"""
import argparse
import os
import pickle
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reloc3r.reloc3r_relpose import Reloc3rRelpose, inference_relpose  # noqa: E402
from reloc3r.datasets.egoexo4d import EgoExo4D  # noqa: E402
from reloc3r.datasets.utils.transforms import ImgNorm  # noqa: E402
from reloc3r.utils.metric import get_rot_err, get_transl_ang_err, error_auc  # noqa: E402
from reloc3r.utils.device import to_numpy  # noqa: E402
from torch.utils.data import DataLoader

from datasets_preprocess.generate_embeddings_slfm_egoexo4d import load_model as load_m2b_model, embed_clip  # noqa: E402

VAL_PICKLE = "/vision/vision_data_2/egoexo4d_audio/camera_pose_audio_data/val_frame_pairs_list.pickle"
DOA_ROOT = "/vision/vision_data_2/egoexo4d_audio/camera_pose_audio_data/doa"
SOUND_ROOT = "/vision/vision_data_2/egoexo4d_audio/camera_pose_audio_data/audio"


def sample_val_pairs(n_samples, seed=777):
    with open(VAL_PICKLE, "rb") as f:
        pairs = pickle.load(f)
    rng = random.Random(seed)
    idx = rng.sample(range(len(pairs)), min(n_samples, len(pairs)))
    return [pairs[i] for i in idx]


def build_audio_lookup(pairs, m2b_model, device, sound_size=1000):
    """(take, frame_id) -> 1384-dim float32 tensor (1024 SLfM embed + 360 DOA)."""
    unique = set()
    for p in pairs:
        unique.add((p['take_name'], p['source_frame_id']))
        unique.add((p['take_name'], p['target_frame_id']))

    lookup = {}
    missing = 0
    for take, frame_id in sorted(unique):
        wav_path = os.path.join(SOUND_ROOT, take, "sound", f"{frame_id:06d}_duration_{sound_size}ms.wav")
        doa_path = os.path.join(DOA_ROOT, take, f"doa_{frame_id:06d}_duration_{sound_size}ms.npy")
        if not (os.path.exists(wav_path) and os.path.exists(doa_path)):
            missing += 1
            continue
        emb = embed_clip(m2b_model, wav_path, device)  # (1024,) numpy float32
        doa = np.load(doa_path).astype(np.float32).squeeze()  # (360,) numpy float32
        lookup[(take, frame_id)] = torch.from_numpy(np.concatenate([emb, doa], axis=0))  # (1384,)
    print(f"Audio lookup built: {len(lookup)} frames ({missing} missing wav/doa, skipped)")
    return lookup


def attach_audio_embedding(view, lookup, frame_id_key):
    """Return a shallow-copied view dict with 'audio_embedding' injected, dropping
    any pairs whose frame is missing from the lookup (returns None for those)."""
    take_names = view['take_name']
    frame_ids = view[frame_id_key].tolist()
    embs = []
    for t, f in zip(take_names, frame_ids):
        if (t, f) not in lookup:
            return None
        embs.append(lookup[(t, f)])
    new_view = dict(view)
    new_view['audio_embedding'] = torch.stack(embs, dim=0)
    return new_view


def load_full_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt['model'] if 'model' in ckpt else ckpt
    res = model.load_state_dict(sd, strict=False)
    print(f"Loaded {ckpt_path}: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}")
    return model


def evaluate(model, loader, device, lookup=None, frame_id_key_pair=('source_frame_id', 'target_frame_id')):
    rerrs, terrs = [], []
    n_skipped = 0
    with torch.no_grad():
        for view1, view2 in loader:
            if lookup is not None:
                v1 = attach_audio_embedding(view1, lookup, frame_id_key_pair[0])
                v2 = attach_audio_embedding(view2, lookup, frame_id_key_pair[1])
                if v1 is None or v2 is None:
                    n_skipped += view1['img'].shape[0]
                    continue
                batch = (v1, v2)
            else:
                batch = (view1, view2)

            pose2to1 = inference_relpose(batch, model, device, use_amp=True)
            gt_pose2to1 = torch.inverse(view1['camera_pose'].to(device).float()) @ view2['camera_pose'].to(device).float()

            R_prd = pose2to1[:, 0:3, 0:3]
            t_prd = pose2to1[:, 0:3, 3]
            for sid in range(R_prd.shape[0]):
                rerr = get_rot_err(to_numpy(R_prd[sid]), to_numpy(gt_pose2to1[sid, 0:3, 0:3]))
                transl = to_numpy(t_prd[sid])
                gt_transl = to_numpy(gt_pose2to1[sid, 0:3, 3])
                transl_dir = transl / (np.linalg.norm(transl) + 1e-8)
                gt_transl_dir = gt_transl / (np.linalg.norm(gt_transl) + 1e-8)
                terr = get_transl_ang_err(transl_dir, gt_transl_dir)
                rerrs.append(rerr)
                terrs.append(terr)
    if n_skipped:
        print(f"Skipped {n_skipped} samples with no audio embedding/DOA available")
    return np.array(rerrs), np.array(terrs)


def report(name, rerrs, terrs):
    print(f"\n=== {name} (n={len(rerrs)}) ===")
    print("Mean rotation error (deg):", np.mean(rerrs))
    print("Median rotation error (deg):", np.median(rerrs))
    print("Mean translation angular error (deg):", np.mean(terrs))
    print("Median translation angular error (deg):", np.median(terrs))
    print("Total AUCs:", error_auc(rerrs, terrs, thresholds=[5, 10, 20]))
    print("Rotation-only AUCs:", error_auc(rerrs, np.zeros_like(terrs), thresholds=[5, 10, 20]))
    print("Translation-only AUCs:", error_auc(np.zeros_like(rerrs), terrs, thresholds=[5, 10, 20]))


def main():
    parser = argparse.ArgumentParser(description="Quick vision-only vs audio+DOA comparison on the val split")
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--vision_only_ckpt', type=str, default='checkpoints/vision_only_quick_ft/checkpoint-best.pth')
    parser.add_argument('--mine_ckpt', type=str,
                         default='checkpoints/_egoexo4d-vision_slfm_embed_and_doa_360_1000ms-512_/checkpoint-best.pth')
    parser.add_argument('--m2b_ckpt', type=str, default='checkpoints/slfm_m2b_egoexo4d/checkpoint-best.pth')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    pairs = sample_val_pairs(args.n_samples, seed=args.seed)
    print(f"Sampled {len(pairs)} val pairs from {VAL_PICKLE}")

    dataset = EgoExo4D(split='val', use_img=True, resolution=(args.img_size, args.img_size),
                        transform=ImgNorm, seed=args.seed)
    dataset.frame_pairs_list = pairs
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, drop_last=False)

    # ---- vision-only baseline ----
    vision_model = Reloc3rRelpose(img_size=args.img_size, has_audio=False, has_audio_embedding=False).to(device)
    load_full_checkpoint(vision_model, args.vision_only_ckpt, device)
    vision_model.eval()
    rerrs_v, terrs_v = evaluate(vision_model, loader, device, lookup=None)
    report("vision_only (quick fine-tune)", rerrs_v, terrs_v)
    del vision_model
    torch.cuda.empty_cache()

    # ---- mine: shipped audio+DOA checkpoint w/ reconstructed embeddings ----
    m2b_model = load_m2b_model(args.m2b_ckpt, device)
    lookup = build_audio_lookup(pairs, m2b_model, device)
    del m2b_model
    torch.cuda.empty_cache()

    mine_model = Reloc3rRelpose(img_size=args.img_size, has_audio=True, has_audio_embedding=True,
                                 use_doa_and_embed=True).to(device)
    load_full_checkpoint(mine_model, args.mine_ckpt, device)
    mine_model.eval()
    rerrs_m, terrs_m = evaluate(mine_model, loader, device, lookup=lookup)
    report("mine (shipped audio+DOA ckpt, reconstructed SLfM embeddings)", rerrs_m, terrs_m)


if __name__ == '__main__':
    main()
