"""
Regenerate the `embeddings_doa_1000ms_alt/` cache directory at the repo root.

This directory is read by `reloc3r/datasets/egoexo4d.py` (use_slfm_and_doa branch)
as: torch.load(f"embeddings_doa_1000ms_alt/{take_name}/doa_{label}.pt"), where
label = f"{take_name}_{frame_id:06d}".

The underlying 360-dim DOA azimuth spectra (pyroomacoustics NormMUSIC, float64,
shape (360,)) are already cached as .npy files under:
    DATA_ROOT/doa/{take_name}/doa_{frame_id:06d}_duration_1000ms.npy

This script:
  1. Loads the train/val frame-pairs pickles.
  2. Builds the set of unique (take_name, frame_id) pairs referenced as either
     source or target across both splits.
  3. For each unique label, copies/converts the corresponding .npy -> .pt
     (float32 torch tensor) under embeddings_doa_1000ms_alt/{take_name}/.
  4. Reports successes/missing and logs missing take names to a file.
"""

import argparse
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from tqdm.auto import tqdm

DATA_ROOT = "/vision/vision_data_2/egoexo4d_audio/camera_pose_audio_data/"
TRAIN_PAIRS_PICKLE = os.path.join(DATA_ROOT, "train_frame_pairs_list.pickle")
VAL_PAIRS_PICKLE = os.path.join(DATA_ROOT, "val_frame_pairs_list.pickle")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "embeddings_doa_1000ms_alt")
MISSING_LOG = os.path.join(REPO_ROOT, "embeddings_doa_1000ms_alt_missing.log")


def load_pairs(pickle_path):
    with open(pickle_path, "rb") as f:
        return pickle.load(f)


def collect_unique_labels():
    """Return sorted list of (take_name, frame_id) tuples needed as source or target."""
    unique = set()
    for pickle_path in (TRAIN_PAIRS_PICKLE, VAL_PAIRS_PICKLE):
        pairs = load_pairs(pickle_path)
        for entry in pairs:
            take_name = entry["take_name"]
            unique.add((take_name, int(entry["source_frame_id"])))
            unique.add((take_name, int(entry["target_frame_id"])))
    return sorted(unique)


def process_one(item):
    """Process a single (take_name, frame_id) tuple.

    Returns (label, take_name, status) where status is "ok", "missing", or an
    error string.
    """
    take_name, frame_id = item
    label = f"{take_name}_{frame_id:06d}"
    src = os.path.join(DATA_ROOT, "doa", take_name, f"doa_{frame_id:06d}_duration_1000ms.npy")
    if not os.path.exists(src):
        return label, take_name, "missing"

    dest_dir = os.path.join(OUT_DIR, take_name)
    dest = os.path.join(dest_dir, f"doa_{label}.pt")

    try:
        arr = np.load(src).astype(np.float32)
        assert arr.shape == (360,), f"unexpected shape {arr.shape} for {src}"
        os.makedirs(dest_dir, exist_ok=True)
        torch.save(torch.from_numpy(arr), dest)
        return label, take_name, "ok"
    except Exception as e:  # pragma: no cover - defensive, reported not raised
        return label, take_name, f"error: {e}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N unique labels (for testing).")
    parser.add_argument("--workers", type=int, default=1,
                         help="Number of worker processes. 1 = plain sequential loop.")
    args = parser.parse_args()

    print(f"Loading pair pickles from {DATA_ROOT} ...")
    unique_items = collect_unique_labels()
    print(f"Found {len(unique_items)} unique (take_name, frame_id) pairs across train+val.")

    if args.limit is not None:
        unique_items = unique_items[: args.limit]
        print(f"--limit set: processing only {len(unique_items)} items.")

    os.makedirs(OUT_DIR, exist_ok=True)

    ok_count = 0
    missing_labels = []
    error_labels = []
    missing_takes = set()

    if args.workers <= 1:
        for item in tqdm(unique_items, desc="Generating DOA embeddings"):
            label, take_name, status = process_one(item)
            if status == "ok":
                ok_count += 1
            elif status == "missing":
                missing_labels.append(label)
                missing_takes.add(take_name)
            else:
                error_labels.append((label, status))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(process_one, item) for item in unique_items]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Generating DOA embeddings"):
                label, take_name, status = fut.result()
                if status == "ok":
                    ok_count += 1
                elif status == "missing":
                    missing_labels.append(label)
                    missing_takes.add(take_name)
                else:
                    error_labels.append((label, status))

    with open(MISSING_LOG, "w") as f:
        f.write(f"# {len(missing_labels)} missing labels, {len(missing_takes)} unique take names\n")
        f.write("# Unique take names with at least one missing source .npy:\n")
        for t in sorted(missing_takes):
            f.write(f"{t}\n")
        f.write("\n# All missing labels:\n")
        for label in missing_labels:
            f.write(f"{label}\n")
        if error_labels:
            f.write("\n# Errors (unexpected):\n")
            for label, err in error_labels:
                f.write(f"{label}\t{err}\n")

    print("=" * 60)
    print(f"Total unique labels needed: {len(unique_items)}")
    print(f"Succeeded:                  {ok_count}")
    print(f"Missing source .npy:        {len(missing_labels)} (across {len(missing_takes)} takes)")
    print(f"Errors:                     {len(error_labels)}")
    print(f"Missing/error log written to: {MISSING_LOG}")
    print(f"Output directory: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
