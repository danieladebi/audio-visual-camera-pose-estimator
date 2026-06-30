import json
import os
import soundfile as sf
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

if __name__ == "__main__":
    methods_measured_train = ["vision_only_train", "audio_only_1000ms_train"] 
    methods_measured_test = ["vision_only", "audio_only_1000ms"]

    
    ground_truth_pose_diffs_train_file = "ground_truth_pose_diffs_train.json"
    with open(ground_truth_pose_diffs_train_file, "r") as f:
        ground_truth_pose_diffs_train = json.load(f)
        ground_truth_pose_diffs_train = {eval(k): v for k, v in ground_truth_pose_diffs_train.items()}

    ground_truth_pose_diffs_test_file = "ground_truth_pose_diffs.json"
    with open(ground_truth_pose_diffs_test_file, "r") as f:
        ground_truth_pose_diffs_test = json.load(f)
        ground_truth_pose_diffs_test = {eval(k): v for k, v in ground_truth_pose_diffs_test.items()}

    save_dir_train = f"video_prediction_errs_{'_'.join(methods_measured_train)}"
    with open(os.path.join(save_dir_train, "max_errors_per_vid_train.json"), "r") as f:
        max_error_per_vid_train = json.load(f)
        max_error_per_vid_train = {k: {eval(kk): vv for kk, vv in v.items()} for k, v in max_error_per_vid_train.items()}

    save_dir_test = f"video_prediction_errs_{'_'.join(methods_measured_test)}"
    with open(os.path.join(save_dir_test, "max_errors_per_vid.json"), "r") as f:
        max_error_per_vid_test = json.load(f)
        max_error_per_vid_test = {k: {eval(kk): vv for kk, vv in v.items()} for k, v in max_error_per_vid_test.items()}

    # TODO: load translation and rotation errors too
    camera_pose_audio_dir = "/vision/vision_data_2/egoexo4d_audio/camera_pose_audio_data/"
    frames_dir = os.path.join(camera_pose_audio_dir, "frames")
    audio_dir = os.path.join(camera_pose_audio_dir, "audio")
    sound_size = 1000  # in ms
    def _load_pair(frame_pair_info):
        vid_name, source_frame_id, target_frame_id = frame_pair_info
        src_audio_file = os.path.join(
            audio_dir, vid_name, "sound",
            f"{source_frame_id:06d}_duration_{sound_size}ms.wav"
        )
        tgt_audio_file = os.path.join(
            audio_dir, vid_name, "sound",
            f"{target_frame_id:06d}_duration_{sound_size}ms.wav"
        )

        if not (os.path.exists(src_audio_file) and os.path.exists(tgt_audio_file)):
            return (frame_pair_info, None, f"Missing: {src_audio_file} or {tgt_audio_file}")

        try:
            source_waveform, sample_rate_src = sf.read(src_audio_file, always_2d=False)
            target_waveform, sample_rate_tgt = sf.read(tgt_audio_file, always_2d=False)
            if sample_rate_src != sample_rate_tgt:
                return (frame_pair_info, None, f"Sample rate mismatch {sample_rate_src} != {sample_rate_tgt}")
            return (frame_pair_info, (source_waveform, target_waveform, sample_rate_src), None)
        except Exception as e:
            return (frame_pair_info, None, str(e))

    train_frame_pairs = list(ground_truth_pose_diffs_train.keys())
    test_frame_pairs = list(ground_truth_pose_diffs_test.keys())
    max_workers = min(8, (os.cpu_count() or 1) * 4)

    loaded_audio_pairs_train = {}
    loaded_audio_pairs_test = {}
    loaded_audio_pairs = {}  # combined (for backward compatibility)
    errors = []

    # Load train audio
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_load_pair, k): k for k in train_frame_pairs}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Loading audio (train)"):
            k, result, err = fut.result()
            if err is not None:
                errors.append((k, err))
            else:
                loaded_audio_pairs_train[k] = result
                loaded_audio_pairs[k] = result

    # Load test audio
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_load_pair, k): k for k in test_frame_pairs}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Loading audio (test)"):
            k, result, err = fut.result()
            if err is not None:
                errors.append((k, err))
            else:
                loaded_audio_pairs_test[k] = result
                loaded_audio_pairs[k] = result

    if errors:
        # Fail fast (or log). Raise first error to keep behavior similar to original.
        first_err = errors[0]
        raise Exception(f"Audio loading errors encountered. Example: {first_err} (total {len(errors)})")