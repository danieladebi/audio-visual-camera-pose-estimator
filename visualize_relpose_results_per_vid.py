import json
from pprint import pprint
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from matplotlib import colors as mcolors
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
from moviepy.editor import TextClip, CompositeVideoClip
from matplotlib import patches
from moviepy.editor import ImageSequenceClip
import matplotlib
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.editor import concatenate_audioclips, concatenate_videoclips
import tempfile
from moviepy.editor import VideoFileClip
import pyroomacoustics as pra
from scipy.spatial.distance import cosine
import soundfile as sf
import ast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import joblib
from sklearn.neural_network import MLPClassifier
from concurrent.futures import ThreadPoolExecutor, as_completed

video_path = "/vision/vision_data_2/EgoExo4D_public_v1/downscaled_takes/448"

create_plots = False
generate_videos = False
calc_mag_residuals = False
calc_signal_to_noise = False
train_policy = True

def error_auc(rError, tErrors, thresholds):
    """
    Args:
        Error (list): [N,]
        tErrors (list): [N,]
        thresholds (list)
    """
    error_matrix = np.concatenate((rError[:, None], tErrors[:, None]), axis=1)
    max_errors = np.max(error_matrix, axis=1)
    errors = [0] + sorted(list(max_errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []
    # thresholds = [5, 10, 20, 30]
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index-1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)

    return {f'auc@{t}': auc for t, auc in zip(thresholds, aucs)}

if __name__ == "__main__":
    frame_pairs_vision_only = np.load("frame_pair_pose_errors_vision_only.npy", allow_pickle=True).item()
    frame_pairs_audio_only_60ms = np.load("frame_pair_pose_errors_audio_only_60ms.npy", allow_pickle=True).item()
    frame_pairs_audio_only_500ms = np.load("frame_pair_pose_errors_audio_only_500ms.npy", allow_pickle=True).item()
    frame_pairs_audio_only_1000ms = np.load("frame_pair_pose_errors_audio_only_1000ms.npy", allow_pickle=True).item()
    frame_pairs_doa_cp_60ms = np.load("frame_pair_pose_errors_doa_cp_60ms.npy", allow_pickle=True).item()
    frame_pairs_doa_cp_500ms = np.load("frame_pair_pose_errors_doa_cp_500ms.npy", allow_pickle=True).item()
    frame_pairs_doa_cp_1000ms = np.load("frame_pair_pose_errors_doa_cp_1000ms.npy", allow_pickle=True).item()
    frame_pairs_baseline_mean = np.load("frame_pair_pose_errors_baseline_mean.npy", allow_pickle=True).item()

    frame_pairs_audio_only_60ms_train = np.load("frame_pair_pose_errors_audio_only_60ms_TRAIN.npy", allow_pickle=True).item()
    frame_pairs_audio_only_500ms_train = np.load("frame_pair_pose_errors_audio_only_500ms_TRAIN.npy", allow_pickle=True).item()
    frame_pairs_audio_only_1000ms_train = np.load("frame_pair_pose_errors_audio_only_1000ms_TRAIN.npy", allow_pickle=True).item()
    frame_pairs_vision_only_train = np.load("frame_pair_pose_errors_vision_only_TRAIN.npy", allow_pickle=True).item()
    frame_pairs_doa_cp_1000ms_train = np.load("frame_pair_pose_errors_doa_cp_1000ms_TRAIN.npy", allow_pickle=True).item()

    frame_pair_results = {
        "vision_only": frame_pairs_vision_only,
        "audio_only_60ms": frame_pairs_audio_only_60ms,
        "audio_only_500ms": frame_pairs_audio_only_500ms,
        "audio_only_1000ms": frame_pairs_audio_only_1000ms,
        "doa_cp_60ms": frame_pairs_doa_cp_60ms,
        "doa_cp_500ms": frame_pairs_doa_cp_500ms,
        "doa_cp_1000ms": frame_pairs_doa_cp_1000ms,
        "baseline_mean": frame_pairs_baseline_mean,
        "vision_only_train": frame_pairs_vision_only_train,
        "audio_only_60ms_train": frame_pairs_audio_only_60ms_train,
        "audio_only_500ms_train": frame_pairs_audio_only_500ms_train,
        "audio_only_1000ms_train": frame_pairs_audio_only_1000ms_train,
        'doa_cp_1000ms_train': frame_pairs_doa_cp_1000ms_train
    }

    frame_pairs_per_vid = {}

    methods_measured = ["vision_only", "doa_cp_1000ms"] 
    for method in frame_pair_results:
      #  print(f"Results for {method}:")
        if method in methods_measured:
            results = frame_pair_results[method]
            for frame_pair in results:
                # print(f"Frame pair: {frame_pair}")
                # raise Exception(type(frame_pair))
                vid_name = frame_pair[0]
                frame_pair_indices = frame_pair[1:]
                frame_pairs_per_vid.setdefault(vid_name, []).append(frame_pair_indices)

                #print(frame_pair, results[frame_pair])
            break

   # pprint(frame_pairs_per_vid.keys())

    rotation_errors_per_vid = {vid_name: {} for vid_name in frame_pairs_per_vid}
    translation_errors_per_vid = {vid_name: {} for vid_name in frame_pairs_per_vid}
    max_error_per_vid = {vid_name: {} for vid_name in frame_pairs_per_vid}

    for vid_name, frame_indices in tqdm(frame_pairs_per_vid.items()):
        for method in methods_measured:
            results = frame_pair_results[method]
            for frame_pair in frame_indices:
                frame_pair_key = (vid_name, frame_pair[0], frame_pair[1])

                # add method to dictionary of frame_pairs if not already present
                if frame_pair not in rotation_errors_per_vid[vid_name]:
                    rotation_errors_per_vid[vid_name][frame_pair] = {}
                    translation_errors_per_vid[vid_name][frame_pair] = {}
                    max_error_per_vid[vid_name][frame_pair] = {}

                rotation_errors_per_vid[vid_name][frame_pair][method] = float(frame_pair_results[method][frame_pair_key]["rotation_error"])
                translation_errors_per_vid[vid_name][frame_pair][method] = float(frame_pair_results[method][frame_pair_key]["translation_error"])
                max_error_per_vid[vid_name][frame_pair][method] = max(rotation_errors_per_vid[vid_name][frame_pair][method], translation_errors_per_vid[vid_name][frame_pair][method])

   # pprint(max_error_per_vid)
    save_dir = f"video_prediction_errs_{'_'.join(methods_measured)}"
    os.makedirs(save_dir, exist_ok=True)
    def convert_keys_to_strings(data):
        if isinstance(data, dict):
            return {str(k): convert_keys_to_strings(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [convert_keys_to_strings(i) for i in data]
        else:
            return data

    with open(os.path.join(save_dir, "rotation_errors_per_vid.json"), "w") as f:
        json.dump(convert_keys_to_strings(rotation_errors_per_vid), f, indent=4)
    with open(os.path.join(save_dir, "translation_errors_per_vid.json"), "w") as f:
        json.dump(convert_keys_to_strings(translation_errors_per_vid), f, indent=4)
    with open(os.path.join(save_dir, "max_errors_per_vid.json"), "w") as f:
        json.dump(convert_keys_to_strings(max_error_per_vid), f, indent=4) 
    
    # Count how often vision beats audio and vice versa (using current methods_measured)
    vision_method = next((m for m in methods_measured if "vision" in m.lower()), None)
    audio_method = next((m for m in methods_measured if "audio" in m.lower() or "doa" in m.lower()), None)

    if vision_method is None or audio_method is None:
        print(f"Cannot determine vision/audio methods from methods_measured={methods_measured}")
    else:
        vision_better = 0
        audio_better = 0
        ties = 0
        skipped = 0
        compared = 0

        for vid_name, pairs in max_error_per_vid.items():
            for pair, errs in pairs.items():
                if vision_method not in errs or audio_method not in errs:
                    skipped += 1
                    continue
                v_err = float(errs[vision_method])
                a_err = float(errs[audio_method])
                compared += 1
                if v_err < a_err:
                    vision_better += 1
                elif a_err < v_err:
                    audio_better += 1
                else:
                    ties += 1

        print(f"Compared frame pairs: {compared}")
        print(f"Vision better: {vision_better}")
        print(f"Audio better: {audio_better}")
        print(f"Ties: {ties}")
        print(f"Skipped (missing either method): {skipped}")

    
    # ground_truth_pose_diffs_file = "ground_truth_pose_diffs_train.json"
    # with open(ground_truth_pose_diffs_file, "r") as f:
    #     ground_truth_pose_diffs = json.load(f)
    #     ground_truth_pose_diffs = {eval(k): v for k, v in ground_truth_pose_diffs.items()}

    raise Exception("STOP", methods_measured)

    #raise Exception(ground_truth_pose_diffs)

    # scatter plot: x = GT rotation, y = GT translation, color by winning method
    # TODO: change circle alpha so that more opaque cirle means larger difference in error. But also keep in mind some differences are very large, circles shouldn't get too big otherwise plot will not be legible.
    if create_plots:
        for vid_name, pairs_dict in tqdm(max_error_per_vid_test.items()):
            xs, ys, winners, margins = [], [], [], []
            for frame_pair, method_errors in pairs_dict.items():
                key = (vid_name, frame_pair[0], frame_pair[1])
                gt = ground_truth_pose_diffs.get(key, None)
                if gt is None:
                    continue
                gt_rot = gt.get("gt_rot", None)
                gt_trans = gt.get("gt_trans_mag", None)
                if gt_rot is None or gt_trans is None:
                    continue
                try:
                    xr = float(gt_rot)
                    yt = float(gt_trans)
                except Exception:
                    continue
                # compute margin = second_best_error - best_error (how much better the winner is)
                errors = sorted([float(e) for e in method_errors.values()])
                margin = errors[1] - errors[0] if len(errors) > 1 else 0.0

                best_method = min(method_errors, key=method_errors.get)
                xs.append(xr)
                ys.append(yt)
                winners.append(best_method)
                margins.append(margin)

            if not xs:
                print(f"No valid GT points for {vid_name}, skipping plot.")
                continue

            # normalize margins to alpha in [min_alpha, max_alpha] (larger margin -> more opaque)
            margins_arr = np.array(margins, dtype=float)
            min_alpha, max_alpha = 0.15, 1.0
            if margins_arr.size == 0:
                alphas = np.array([])
            else:
                mn, mx = margins_arr.min(), margins_arr.max()
                if mx <= mn:
                    alphas = np.full_like(margins_arr, (min_alpha + max_alpha) / 2.0)
                else:
                    scaled = (margins_arr - mn) / (mx - mn)
                    alphas = min_alpha + scaled * (max_alpha - min_alpha)

            # color mapping for methods (supports arbitrary number of methods)
            cmap = plt.get_cmap("tab10")
            # use blue and red for methods (repeats if more than two methods)
            base_palette = ["blue", "red"]
            method_colors = {m: base_palette[i % len(base_palette)] for i, m in enumerate(methods_measured)}
            base_colors = [method_colors.get(m, "gray") for m in winners]

            # construct RGBA colors with per-point alpha
            rgba_colors = []
            for base_c, a in zip(base_colors, alphas):
                rgba = mcolors.to_rgba(base_c)
                rgba_colors.append((rgba[0], rgba[1], rgba[2], float(a)))

            plt.figure(figsize=(8, 6))
            plt.scatter(xs, ys, c=rgba_colors, edgecolor="k")

            # legend entries for all measured methods
            handles = [
                plt.Line2D([0], [0], marker="o", color="w", label=m,
                        markerfacecolor=method_colors.get(m, "gray"),
                        markeredgecolor="k", markersize=8)
                for m in methods_measured
            ]
            plt.legend(handles=handles, title="Best method")
            plt.xlabel("GT rotation (degrees)")
            plt.ylabel("GT translation magnitude")
            plt.title(f"Best method by GT rotation vs GT translation — {vid_name}")
            plt.grid(True, linestyle="--", alpha=0.3)
            plt.tight_layout()

            out_dir = f"rot_vs_trans_per_video_alpha_{'_'.join(methods_measured)}"
            os.makedirs(out_dir, exist_ok=True)
            safe_name = re.sub(r'[^A-Za-z0-9._-]', '_', vid_name)
            plt.savefig(os.path.join(out_dir, f"{safe_name}_gt_rot_vs_trans_alpha_method_wins.png"))
            plt.close()


    # TODO: generate videos that show frames AND audio 
    # METHOD SHOULD PULL EACH FRAME INDEX (source frame and target frame) AND PLAY AUDIO CLIP FROM SOURCE FRAME AND AUDIO FRAME

    camera_pose_audio_dir = "/vision/vision_data_2/egoexo4d_audio/camera_pose_audio_data/"
    frames_dir = os.path.join(camera_pose_audio_dir, "frames")
    output_audio_dir = os.path.join(camera_pose_audio_dir, "audio")

    if "60ms" in methods_measured[1]:
        sound_size = 60
    elif "500ms" in methods_measured[1]:
        sound_size = 500
    elif "1000ms" in methods_measured[1]:
        sound_size = 1000
    else:
        raise Exception("Cannot determine sound size from method name")

    visualization_save_dir = f"visualization_preds_{'_'.join(methods_measured)}"
    matplotlib.use("Agg")  # headless

    os.makedirs(visualization_save_dir, exist_ok=True)

    # Assign consistent colors per method (extend / modify as needed)
    default_palette = ["tab:blue", "tab:red", "tab:green", "tab:orange", "tab:purple", "tab:brown"]
    method_colors = {m: default_palette[i % len(default_palette)] for i, m in enumerate(methods_measured)}

    fps = 24
    post_audio_hold = 0.4  # seconds to keep frame visible after audio
    highlight_color = "yellow"
    target_hold_highlight = True  # if True, highlight target after audio completes


    if generate_videos:
        # One output video per vid_name (concatenating all frame_pairs)
        for vid_name in tqdm(max_error_per_vid_test, position=0):
            frame_pairs = sorted(max_error_per_vid_test[vid_name].keys())
            segment_clips = []

            # Collect ALL video (silent) segments first, then add concatenated audio at end
            segment_audios = []  # store extended_audio clips in order (one per frame_pair)

            # debug_audio_dir = os.path.join(visualization_save_dir, "debug_audio", vid_name)
            # os.makedirs(debug_audio_dir, exist_ok=True)

            for frame_pair in tqdm(frame_pairs, position=1, leave=False):
                source_frame_id, target_frame_id = frame_pair
                source_frame_file = os.path.join(frames_dir, vid_name, f"{source_frame_id:06d}.jpg")
                target_frame_file = os.path.join(frames_dir, vid_name, f"{target_frame_id:06d}.jpg")

                src_audio_file = os.path.join(
                    output_audio_dir, vid_name, "sound",
                    f"{source_frame_id:06d}_duration_{sound_size}ms.wav"
                )
                tgt_audio_file = os.path.join(
                    output_audio_dir, vid_name, "sound",
                    f"{target_frame_id:06d}_duration_{sound_size}ms.wav"
                )

                if not (os.path.exists(source_frame_file)
                        and os.path.exists(target_frame_file)
                        and os.path.exists(src_audio_file)):
                    continue

                method_errors = max_error_per_vid_test.get(vid_name, {}).get(frame_pair, {})
                available = {m: method_errors[m] for m in methods_measured if m in method_errors}
                if not available:
                    continue
                best_method = min(available, key=available.get)
                best_max_err = available[best_method]

                try:
                    best_rot_err = rotation_errors_per_vid_test[vid_name][frame_pair][best_method]
                    best_trans_err = translation_errors_per_vid_test[vid_name][frame_pair][best_method]
                except KeyError:
                    best_rot_err = float('nan')
                    best_trans_err = float('nan')

                # Load source audio
                try:
                    src_audio_clip = AudioFileClip(src_audio_file)
                except Exception:
                    continue  # skip bad audio

                target_audio_available = os.path.exists(tgt_audio_file)
                tgt_audio_clip = None
                if target_audio_available:
                    try:
                        tgt_audio_clip = AudioFileClip(tgt_audio_file)
                    except Exception:
                        tgt_audio_clip = None
                        target_audio_available = False

                if target_audio_available:
                    src_dur = float(src_audio_clip.duration)
                    tgt_dur = float(tgt_audio_clip.duration)
                    sr = src_audio_clip.fps
                    n_channels = src_audio_clip.nchannels
                    silence_samples = max(1, int(post_audio_hold * sr))
                    silence_array = np.zeros((silence_samples, n_channels), dtype=np.float32)
                    silence_clip = AudioArrayClip(silence_array, fps=sr)
                    extended_audio = concatenate_audioclips([src_audio_clip, tgt_audio_clip, silence_clip])
                    total_duration = src_dur + tgt_dur + post_audio_hold
                    dual_audio = True
                else:
                    audio_dur = float(src_audio_clip.duration)
                    sr = src_audio_clip.fps
                    n_channels = src_audio_clip.nchannels
                    silence_samples = max(1, int(post_audio_hold * sr))
                    silence_array = np.zeros((silence_samples, n_channels), dtype=np.float32)
                    silence_clip = AudioArrayClip(silence_array, fps=sr)
                    extended_audio = concatenate_audioclips([src_audio_clip, silence_clip])
                    total_duration = audio_dur + post_audio_hold
                    src_dur = audio_dur
                    tgt_dur = 0.0
                    dual_audio = False

                # # Save debug audio to verify it's not mute
                # debug_audio_path = os.path.join(
                #     debug_audio_dir,
                #     f"{vid_name}_{source_frame_id:06d}_{target_frame_id:06d}_extended.wav"
                # )
                # try:
                #     extended_audio.write_audiofile(debug_audio_path, fps=sr, verbose=False, logger=None)
                # except Exception as e:
                #     print(f"Failed to write debug audio {debug_audio_path}: {e}")

                # Do NOT close src_audio_clip / tgt_audio_clip here; extended_audio still references them.
                # They will be closed later in the cleanup loop after final concatenation.
                # Build images / visualization (silent frames)
                src_img = plt.imread(source_frame_file)
                tgt_img = plt.imread(target_frame_file)

                fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                ax_src, ax_tgt = axes
                ax_src.imshow(src_img)
                ax_tgt.imshow(tgt_img)
                for ax in axes:
                    ax.axis("off")

                rect_src = patches.Rectangle(
                    (0, 0), src_img.shape[1], src_img.shape[0],
                    linewidth=6, edgecolor=highlight_color, facecolor='none'
                )
                rect_tgt = patches.Rectangle(
                    (0, 0), tgt_img.shape[1], tgt_img.shape[0],
                    linewidth=6, edgecolor=highlight_color, facecolor='none'
                )
                ax_src.add_patch(rect_src)
                ax_tgt.add_patch(rect_tgt)
                rect_tgt.set_visible(False)

                # TODO: change to ground truth error:
                text_str = (
                    f"Best: {best_method}\n"
                    f"rot={best_rot_err:.3f}  trans={best_trans_err:.3f}  max={best_max_err:.3f}"
                )
                fig.text(
                    0.5, 0.045,
                    text_str,
                    ha="center", va="center",
                    fontsize=12,
                    color=method_colors.get(best_method, "white"),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.5, edgecolor="none")
                )

                n_frames = max(1, int(math.ceil(total_duration * fps)))
                frames = []
                canvas = fig.canvas

                for i in range(n_frames):
                    t = i / fps
                    if dual_audio:
                        if t < src_dur:
                            rect_src.set_visible(True)
                            rect_tgt.set_visible(False)
                        elif t < src_dur + tgt_dur:
                            rect_src.set_visible(False)
                            rect_tgt.set_visible(True)
                        else:
                            if target_hold_highlight:
                                rect_src.set_visible(False)
                                rect_tgt.set_visible(True)
                            else:
                                rect_src.set_visible(False)
                                rect_tgt.set_visible(False)
                    else:
                        if t < src_dur:
                            rect_src.set_visible(True)
                            rect_tgt.set_visible(False)
                        else:
                            if target_hold_highlight:
                                rect_src.set_visible(False)
                                rect_tgt.set_visible(True)
                            else:
                                rect_src.set_visible(False)
                                rect_tgt.set_visible(False)

                    fig.tight_layout(pad=0)
                    canvas.draw()
                    w, h = canvas.get_width_height()
                    frame = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
                    frame = frame[:, :, [1, 2, 3, 0]]  # ARGB -> RGBA
                    frames.append(frame.copy())

                plt.close(fig)

                # Write silent temp video
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_vid_file:
                    temp_vid_path = temp_vid_file.name

                temp_img_seq_clip = ImageSequenceClip(frames, fps=fps).set_duration(total_duration)
                temp_img_seq_clip.write_videofile(
                    temp_vid_path, fps=fps, codec="libx264", audio=False, verbose=False, logger=None
                )
                temp_img_seq_clip.close()

                img_seq_clip = VideoFileClip(temp_vid_path)  # silent segment
                segment_clips.append(img_seq_clip)
                segment_audios.append(extended_audio)

            if not segment_clips:
                print(f"No valid segments for {vid_name}, skipping video.")
                continue

            final_clip = concatenate_videoclips(segment_clips, method="compose")

            # Concatenate all audio AFTER videos are concatenated
            if segment_audios:
                try:
                    final_audio = concatenate_audioclips(segment_audios)
                    final_clip = final_clip.set_audio(final_audio)
                except Exception as e:
                    print(f"Warning: failed to concatenate audio for {vid_name}: {e}")

            out_name = f"{vid_name}_all_pairs_audio_{sound_size}ms.mp4"
            out_path = os.path.join(visualization_save_dir, out_name)
            final_clip.write_videofile(
                out_path
            )
            final_clip.close()

            # Cleanup
            for c in segment_clips:
                try:
                    c.close()
                except Exception:
                    pass
            for ac in segment_audios:
                try:
                    ac.close()
                except Exception:
                    pass

        print("Per-video concatenated visualization videos created.")

    # Create and evaluate policy that chooses between which policy is better. 
    # pprint(max_error_per_vid)
   
    median_rot_threshold = None
    median_trans_mag_threshold = None

    rots_per_vid = {}
    trans_mags_per_vid = {}

    rots = []
    trans_mags = []
    for frame_pair_info, gt_data in ground_truth_pose_diffs.items():
        vid_name, source_frame_id, target_frame_id = frame_pair_info
        gt_rot = gt_data["gt_rot"]
        gt_trans_mag = gt_data["gt_trans_mag"]
        rots.append(gt_rot)
        trans_mags.append(gt_trans_mag)

        if vid_name not in rots_per_vid:
            rots_per_vid[vid_name] = []
        rots_per_vid[vid_name].append(gt_rot)

        if vid_name not in trans_mags_per_vid:
            trans_mags_per_vid[vid_name] = []
        trans_mags_per_vid[vid_name].append(gt_trans_mag)

    median_rot_threshold = float(np.median(np.array(rots)))
    mean_rot_threshold = float(np.mean(np.array(rots)))
    median_trans_mag_threshold = float(np.median(np.array(trans_mags)))
    median_per_video = {vid: float(np.median(np.array(rots_per_vid[vid]))) for vid in rots_per_vid}
    mean_per_video = {vid: float(np.mean(np.array(rots_per_vid[vid]))) for vid in rots_per_vid}

    print(median_rot_threshold, median_trans_mag_threshold)

    policy_decisions = {}
    selected_policy_errors = {methods_measured[0]: [], methods_measured[1]: []}
    selected_policy_rot_errors = {methods_measured[0]: [], methods_measured[1]: []}
    selected_policy_trans_errors = {methods_measured[0]: [], methods_measured[1]: []}

    methods_measured_test = ["vision_only", "audio_only_1000ms"]
    save_dir = f"video_prediction_errs_{'_'.join(methods_measured_test)}"
    with open(os.path.join(save_dir, "max_errors_per_vid.json"), "r") as f:
        max_error_per_vid_test = json.load(f)
        max_error_per_vid_test = {k: {eval(kk): vv for kk, vv in v.items()} for k, v in max_error_per_vid_test.items()}
        
    for frame_pair_info, gt_data in ground_truth_pose_diffs.items():
        vid_name, source_frame_id, target_frame_id = frame_pair_info
        gt_rot = gt_data["gt_rot"]
        rot_err = rotation_errors_per_vid_test[vid_name]
        trans_error = translation_errors_per_vid_test[vid_name] # translation direction, not magnitude

     #   policy_decisions[frame_pair_info] = 1 if (gt_rot >= mean_rot_threshold) else 0 # 1 means audio, 0 means vision # median_rot_threshold
        audio_max_error = max_error_per_vid_test[vid_name][(source_frame_id, target_frame_id)][methods_measured[1]]
        vision_max_error = max_error_per_vid_test[vid_name][(source_frame_id, target_frame_id)][methods_measured[0]]

        audio_rot_error = rotation_errors_per_vid_test[vid_name][(source_frame_id, target_frame_id)][methods_measured[1]]
        vision_rot_error = rotation_errors_per_vid_test[vid_name][(source_frame_id, target_frame_id)][methods_measured[0]]

        audio_trans_error = translation_errors_per_vid_test[vid_name][(source_frame_id, target_frame_id)][methods_measured[1]]
        vision_trans_error = translation_errors_per_vid_test[vid_name][(source_frame_id, target_frame_id)][methods_measured[0]]

        min_audio_error = min(audio_rot_error, audio_trans_error)
        min_vision_error = min(vision_rot_error, vision_trans_error)
        #policy_decisions[frame_pair_info] = 1 if audio_max_error <= vision_max_error else 0
        policy_decisions[frame_pair_info] = 0 if min_audio_error <= min_vision_error else 0

        # OPTIMAL MINIMUMS
        #
        selected_policy_errors[methods_measured[1]].append(min(audio_rot_error, audio_trans_error, vision_rot_error, vision_trans_error))
        selected_policy_rot_errors[methods_measured[1]].append(min(audio_rot_error, vision_rot_error))
        selected_policy_trans_errors[methods_measured[1]].append(min(audio_trans_error, vision_trans_error))

        # # END OPTIMAL MINIMUMS

        # if policy_decisions[frame_pair_info] == 1:
        #     audio_method = next((m for m in methods_measured if m.startswith("audio_only")), "audio_only_1000ms")
        #     selected_policy_errors[methods_measured[1]].append(max_error_per_vid_test[vid_name][(source_frame_id, target_frame_id)][audio_method])
        #     selected_policy_rot_errors[methods_measured[1]].append(rotation_errors_per_vid_test[vid_name][(source_frame_id, target_frame_id)][audio_method])
        #     selected_policy_trans_errors[methods_measured[1]].append(translation_errors_per_vid_test[vid_name][(source_frame_id, target_frame_id)][audio_method])
        # else:
        #     vision_method = next((m for m in methods_measured if m.startswith("vision_only")), "vision_only")
        #     selected_policy_errors[methods_measured[0]].append(max_error_per_vid_test[vid_name][(source_frame_id, target_frame_id)][vision_method])
        #     selected_policy_rot_errors[methods_measured[0]].append(rotation_errors_per_vid_test[vid_name][(source_frame_id, target_frame_id)][vision_method])
        #     selected_policy_trans_errors[methods_measured[0]].append(translation_errors_per_vid_test[vid_name][(source_frame_id, target_frame_id)][vision_method])

    full_policy_error_list = selected_policy_errors[methods_measured[0]] + selected_policy_errors[methods_measured[1]]
    full_policy_error_array = np.array(full_policy_error_list)
    rot_policy_error_array = np.array(selected_policy_rot_errors[methods_measured[0]] + selected_policy_rot_errors[methods_measured[1]])
    trans_policy_error_array = np.array(selected_policy_trans_errors[methods_measured[0]] + selected_policy_trans_errors[methods_measured[1]])

    print("Policy results:")
    print(f"Total frame pairs evaluated: {len(full_policy_error_array)}")
    print(f"Mean max error: {np.mean(full_policy_error_array):.4f}")
    print("Mean rotation error: {:.4f}".format(np.mean(rot_policy_error_array)))
    print("Mean translation error: {:.4f}".format(np.mean(trans_policy_error_array)))

    print(f"Median max error: {np.median(full_policy_error_array):.4f}")
    print("Median rotation error: {:.4f}".format(np.median(rot_policy_error_array)))
    print("Median translation error: {:.4f}".format(np.median(trans_policy_error_array)))

    print("Total AUCs")
    aucs = error_auc(
        rot_policy_error_array,
        trans_policy_error_array,
        [5, 10, 20]
    )
    rot_aucs = error_auc(
        rot_policy_error_array,
        np.zeros_like(rot_policy_error_array),
        [5, 10, 20]
    )
    trans_aucs = error_auc(
        trans_policy_error_array,
        np.zeros_like(trans_policy_error_array),
        [5, 10, 20]
    )
    print("TOTAL AUCs")
    print(aucs)
    print("Rotation AUCs")
    print(rot_aucs)
    print("Translation AUCs")
    print(trans_aucs)
    raise Exception("done")

    residual_mags_per_vid_name = {}
    snrs_per_vid_name = {}
    
    residual_mag_path = f"residual_mags_{'-'.join(methods_measured)}"


    # Plot SNRs per video, colored by which model wins (audio=red, visual=blue)
    snr_dir = f"snrs_per_vid_{'-'.join(methods_measured)}"
    os.makedirs(snr_dir, exist_ok=True)

    if calc_mag_residuals:
        if calc_signal_to_noise or not os.path.exists(os.path.join(residual_mag_path, f"residual_magnitudes_per_vid_{'_'.join(methods_measured)}.json")):
            def _compute_resid_snr(args):
                frame_pair_info, _ = args
                try:
                    vid_name, source_frame_id, target_frame_id = frame_pair_info

                    src_audio_file = os.path.join(
                        output_audio_dir, vid_name, "sound",
                        f"{source_frame_id:06d}_duration_{sound_size}ms.wav"
                    )
                    tgt_audio_file = os.path.join(
                        output_audio_dir, vid_name, "sound",
                        f"{target_frame_id:06d}_duration_{sound_size}ms.wav"
                    )

                    if not (os.path.exists(src_audio_file) and os.path.exists(tgt_audio_file)):
                        return None

                    source_waveform, sample_rate = sf.read(src_audio_file, always_2d=False)
                    target_waveform, sample_rate = sf.read(tgt_audio_file, always_2d=False)

                    nfft = 1023
                    hop = nfft // 4 if sound_size == 60 else nfft // 2

                    source_X = pra.transform.stft.analysis(source_waveform, L=nfft, hop=hop)
                    target_X = pra.transform.stft.analysis(target_waveform, L=nfft, hop=hop)

                    eps = 1e-8
                    source_X_mag = 10 * np.log10(np.abs(source_X) + eps)
                    target_X_mag = 10 * np.log10(np.abs(target_X) + eps)

                    residual_mag = float(np.mean(np.abs(target_X_mag - source_X_mag)))

                    src_snr_arr = np.abs(source_X_mag)
                    tgt_snr_arr = np.abs(target_X_mag)
                    source_snr = float(src_snr_arr.mean() / (src_snr_arr.std() + eps))
                    target_snr = float(tgt_snr_arr.mean() / (tgt_snr_arr.std() + eps))
                    average_snr = float((source_snr + target_snr) / 2.0)

                    return (vid_name, (source_frame_id, target_frame_id), residual_mag, average_snr)
                except Exception:
                    return None

            pairs = list(ground_truth_pose_diffs.items())
            max_workers = 32

            residual_mags_per_vid_name = {}
            snrs_per_vid_name = {}

            with ThreadPoolExecutor(max_workers=max_workers) as ex, tqdm(total=len(pairs)) as pbar:
                futures = [ex.submit(_compute_resid_snr, args) for args in pairs]
                for fut in as_completed(futures):
                    res = fut.result()
                    if res is not None:
                        vid_name, pair, residual_mag, average_snr = res
                        residual_mags_per_vid_name.setdefault(vid_name, {})[pair] = residual_mag
                        snrs_per_vid_name.setdefault(vid_name, {})[pair] = average_snr
                    pbar.update(1)

            os.makedirs(snr_dir, exist_ok=True)
            snrs_per_vid_name = {k: {str(kk): vv for kk, vv in v.items()} for k, v in snrs_per_vid_name.items()}
            with open(os.path.join(snr_dir, f"snrs_per_vid_{'_'.join(methods_measured)}.json"), "w") as f:
                json.dump(snrs_per_vid_name, f, indent=4)

            residual_mags_per_vid_name = {str(k): {str(kk): vv for kk, vv in v.items()} for k, v in residual_mags_per_vid_name.items()}
            os.makedirs(residual_mag_path, exist_ok=True)
            with open(os.path.join(residual_mag_path, f"residual_magnitudes_per_vid_{'_'.join(methods_measured)}.json"), "w") as f:
                json.dump(residual_mags_per_vid_name, f, indent=4)
        else:
            with open(os.path.join(residual_mag_path, f"residual_magnitudes_per_vid_{'_'.join(methods_measured)}.json"), "r") as f:
                residual_mags_per_vid_name = json.load(f)
                residual_mags_per_vid_name = {k: {eval(kk): vv for kk, vv in v.items()} for k, v in residual_mags_per_vid_name.items()}
        
            with open(os.path.join(snr_dir, f"snrs_per_vid_{'_'.join(methods_measured)}.json"), "r") as f:
                snrs_per_vid_name = json.load(f)
                snrs_per_vid_name = {k: {eval(kk): vv for kk, vv in v.items()} for k, v in snrs_per_vid_name.items()}
        # Plot residuals per video, colored by which model wins (audio=red, visual=blue)
        os.makedirs(residual_mag_path, exist_ok=True)

        audio_method = next((m for m in methods_measured if m.startswith("audio_only")), "audio_only_1000ms_train")
        visual_method = "vision_only_train"


        # Load SNRs from disk if present and in-memory map is empty
        snr_json_path = os.path.join(snr_dir, f"snrs_per_vid_{'_'.join(methods_measured)}.json")
        if (not snrs_per_vid_name) and os.path.exists(snr_json_path):
            try:
                with open(snr_json_path, "r") as f:
                    loaded_snrs = json.load(f)
                parsed_snrs = {}
                for vid, pair_map in loaded_snrs.items():
                    new_map = {}
                    for k, v in pair_map.items():
                        pair = None
                        if isinstance(k, (list, tuple)) and len(k) == 2:
                            pair = tuple(k)
                        elif isinstance(k, str):
                            try:
                                parsed = ast.literal_eval(k)
                                if isinstance(parsed, (list, tuple)) and len(parsed) == 2:
                                    pair = tuple(parsed)
                            except Exception:
                                continue
                        if pair is not None:
                            try:
                                new_map[pair] = float(v)
                            except Exception:
                                pass
                    parsed_snrs[vid] = new_map
                snrs_per_vid_name = parsed_snrs
            except Exception:
                pass

        for vid_name, pair_map in snrs_per_vid_name.items():
            xs, ys, colors = [], [], []
            idx = 0

            for k, snr_val in pair_map.items():
            # Normalize frame pair key to a tuple of ints
                pair = None
                if isinstance(k, tuple):
                    pair = k
                elif isinstance(k, list):
                    pair = tuple(k)
                elif isinstance(k, str):
                    try:
                        parsed = ast.literal_eval(k)
                        if isinstance(parsed, (list, tuple)) and len(parsed) == 2:
                            pair = tuple(parsed)
                    except Exception:
                        continue
                else:
                    continue

                if not isinstance(pair, tuple) or len(pair) != 2:
                    continue

                errs = max_error_per_vid_train.get(vid_name, {}).get(pair, {})
                if audio_method not in errs or visual_method not in errs:
                    continue

                color = "red" if errs[audio_method] < errs[visual_method] else "blue"

                xs.append(idx)
                try:
                    ys.append(float(snr_val))
                except Exception:
                    continue
                colors.append(color)
                idx += 1

                if not ys:
                    continue

            plt.figure(figsize=(12, 4))
            plt.scatter(xs, ys, c=colors, s=16, edgecolor="k", linewidths=0.2, alpha=0.9)
            plt.xlabel("Frame pair index")
            plt.ylabel("Average SNR")
            plt.title(f"{vid_name} — SNRs (red=audio better, blue=visual better)")
            handles = [
            plt.Line2D([0], [0], marker="o", color="w", label="Audio better",
                   markerfacecolor="red", markeredgecolor="k", markersize=6),
            plt.Line2D([0], [0], marker="o", color="w", label="Visual better",
                   markerfacecolor="blue", markeredgecolor="k", markersize=6),
            ]
            plt.legend(handles=handles, loc="best", framealpha=0.8)
            plt.grid(True, linestyle="--", alpha=0.3)
            safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", str(vid_name))
            plt.tight_layout()
            plt.savefig(os.path.join(snr_dir, f"{safe_name}_snrs_colored.png"))
            plt.close()

        # TODO: Learn a binary classifier to choose the better method using [SNR, residual_mag] features
        # ALSO TODO: FIX THIS CLASSIFER
        # Build dataset: X = [snr, residual_mag], y = 1 if audio better else 0

    if train_policy:
        train_snrs = json.load(open("snrs_per_vid_vision_only_train-audio_only_1000ms_train/snrs_per_vid_vision_only_train_audio_only_1000ms_train.json"))
        train_residuals = json.load(open("residual_mags_vision_only_train-audio_only_1000ms_train/residual_magnitudes_per_vid_vision_only_train_audio_only_1000ms_train.json"))

        train_snrs = {k: {eval(kk): vv for kk, vv in v.items()} for k, v in train_snrs.items()}
        train_residuals = {k: {eval(kk): vv for kk, vv in v.items()} for k, v in train_residuals.items()}

        test_snrs = json.load(open("snrs_per_vid_vision_only-audio_only_1000ms/snrs_per_vid_vision_only_audio_only_1000ms.json"))
        test_residuals = json.load(open("residual_mags_vision_only-audio_only_1000ms/residual_magnitudes_per_vid_vision_only_audio_only_1000ms.json"))

        test_snrs = {k: {eval(kk): vv for kk, vv in v.items()} for k, v in test_snrs.items()}
        test_residuals = {k: {eval(kk): vv for kk, vv in v.items()} for k, v in test_residuals.items()}

        X_train, y_train = [], []

        visual_method = "vision_only_train"
        audio_method = "audio_only_1000ms_train"
        for vid_name, pair_map in train_snrs.items():
            for pair, snr_val in pair_map.items():
                if vid_name not in train_residuals or pair not in train_residuals[vid_name]:
                    continue
                residual_mag = train_residuals[vid_name][pair]
                errs = max_error_per_vid_train.get(vid_name, {}).get(pair, {})
                if audio_method not in errs or visual_method not in errs:
                    continue
                label = 1 if errs[audio_method] < errs[visual_method] else 0
                try:
                    X_train.append([float(snr_val), float(residual_mag)])
                    y_train.append(label)
                except Exception:
                    continue

        methods_measured_test = ["vision_only", "audio_only_1000ms"]
        save_dir = f"video_prediction_errs_{'_'.join(methods_measured_test)}"
        with open(os.path.join(save_dir, "max_errors_per_vid.json"), "r") as f:
            max_error_per_vid_test = json.load(f)
            max_error_per_vid_test = {k: {eval(kk): vv for kk, vv in v.items()} for k, v in max_error_per_vid_test.items()}

        X_test, y_test = [], []
        visual_method = "vision_only"
        audio_method = "audio_only_1000ms"
        for vid_name, pair_map in test_snrs.items():
            for pair, snr_val in pair_map.items():
                if vid_name not in test_residuals or pair not in test_residuals[vid_name]:
                    continue
                residual_mag = test_residuals[vid_name][pair]
                errs = max_error_per_vid_test.get(vid_name, {}).get(pair, {})
                if audio_method not in errs or visual_method not in errs:
                    continue
                label = 1 if errs[audio_method] < errs[visual_method] else 0
                try:
                    X_test.append([float(snr_val), float(residual_mag)])
                    y_test.append(label)
                except Exception:
                    continue

  ###  raise Exception( len(residual_mags_per_vid_name))

  