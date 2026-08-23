"""
Restore the exact `.wav` files `reloc3r/datasets/egoexo4d.py` expects
(`DATA_ROOT/audio/{take_name}/sound/{frame_id:06d}_duration_1000ms.wav`) from
the FLAC-compressed clips shipped in `camera_pose_audio_data.zip`.

The zip stores these clips as lossless FLAC (~4.4x smaller) instead of raw
WAV to keep the download size reasonable -- see
`build_camera_pose_audio_data_zip.py`. This script decodes them back,
losslessly, to real `.wav` files with the exact filenames the loader needs.

Usage (run once, after unzipping camera_pose_audio_data.zip):
    python datasets_preprocess/decode_camera_pose_audio_flac.py \
        --root /path/to/camera_pose_audio_data
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import soundfile as sf
from tqdm.auto import tqdm


def decode_one(args):
    flac_path, delete_flac = args
    wav_path = flac_path[: -len(".flac")] + ".wav"
    try:
        data, sr = sf.read(flac_path, dtype="int16")
        sf.write(wav_path, data, sr, subtype="PCM_16")
        if delete_flac:
            os.remove(flac_path)
        return flac_path, "ok"
    except Exception as e:  # pragma: no cover
        return flac_path, f"error: {e}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=str, required=True,
                         help="path to the extracted camera_pose_audio_data/ directory")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--delete-flac", action="store_true",
                         help="remove the source .flac after a successful decode (saves disk)")
    args = parser.parse_args()

    audio_root = os.path.join(args.root, "audio")
    flac_paths = []
    for take in sorted(os.listdir(audio_root)):
        sound_dir = os.path.join(audio_root, take, "sound")
        if not os.path.isdir(sound_dir):
            continue
        for fname in os.listdir(sound_dir):
            if fname.endswith(".flac"):
                flac_paths.append(os.path.join(sound_dir, fname))

    print(f"Found {len(flac_paths)} .flac files to decode under {audio_root}")

    ok = 0
    errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(decode_one, (p, args.delete_flac)) for p in flac_paths]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Decoding FLAC -> WAV"):
            path, status = fut.result()
            if status == "ok":
                ok += 1
            else:
                errors.append((path, status))

    print(f"Done. {ok}/{len(flac_paths)} decoded successfully, {len(errors)} errors.")
    for path, status in errors[:20]:
        print(f"  {path}: {status}")


if __name__ == "__main__":
    main()
