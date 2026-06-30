import json
from pprint import pprint
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import ks_2samp
from transformers import pipeline
import soundfile as sf
import librosa
import librosa
from scipy.signal import resample_poly
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModel

keystep_json = json.load(open("/vision/vision_data_2/egoexo4d_audio/annotations/keystep_val.json", "r"))
takes_json = json.load(open("/vision/vision_data_2/EgoExo4D_public_v1/takes.json", "r"))
audio_clip_dir = "/vision/vision_data_2/egoexo4d_audio/camera_pose_audio_data/audio"

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
    vision_only_aucs = json.load(open("per_video_auc_results_vision_only.json", "r"))
    baseline_mean_aucs = json.load(open("per_video_auc_results_baseline_mean.json", "r"))
    audio_60ms_aucs = json.load(open("per_video_auc_results_audio_only_60ms.json", "r"))
    audio_500ms_aucs = json.load(open("per_video_auc_results_audio_only_500ms.json", "r"))
    audio_1000ms_aucs = json.load(open("per_video_auc_results_audio_only_1000ms.json", "r"))
    vision_audio_60ms_aucs = json.load(open("per_video_auc_results_audio_embed_60ms.json", "r"))
    vision_audio_500ms_aucs = json.load(open("per_video_auc_results_audio_embed_500ms.json", "r"))
    vision_audio_1000ms_aucs = json.load(open("per_video_auc_results_audio_embed_1000ms.json", "r"))
    audio_doa_60ms_aucs = json.load(open("per_video_auc_results_doa_cp_60ms.json", "r"))
    audio_doa_500ms_aucs = json.load(open("per_video_auc_results_doa_cp_500ms.json", "r"))
    audio_doa_1000ms_aucs = json.load(open("per_video_auc_results_doa_cp_1000ms.json", "r"))
    vision_audio_doa_60ms_aucs = json.load(open("per_video_auc_results_doa_embed_vision_60ms.json", "r"))
    vision_audio_doa_500ms_aucs = json.load(open("per_video_auc_results_doa_embed_vision_500ms.json", "r"))
    vision_audio_doa_1000ms_aucs = json.load(open("per_video_auc_results_doa_embed_vision_1000ms.json", "r"))
    vision_audio_slfm_1000ms_aucs = json.load(open("per_video_auc_results_slfm_embed_vision_1000ms.json", "r"))
    vision_audio_doa_360_aucs = json.load(open("per_video_auc_results_vision_doa_360.json", "r"))
    vision_audio_slfm_doa_360_aucs = json.load(open("per_video_auc_results_vision_doa_360_slfm_embed.json", "r"))
    auc_results = {
        "audio_doa_60ms": audio_doa_60ms_aucs,
        "audio_doa_500ms": audio_doa_500ms_aucs,
        "audio_doa_1000ms": audio_doa_1000ms_aucs,
        "vision_audio_doa_60ms": vision_audio_doa_60ms_aucs,
        "vision_audio_doa_500ms": vision_audio_doa_500ms_aucs,
        "vision_audio_doa_1000ms": vision_audio_doa_1000ms_aucs,
        "vision_audio_60ms": vision_audio_60ms_aucs,
        "vision_audio_500ms": vision_audio_500ms_aucs,
        "vision_audio_1000ms": vision_audio_1000ms_aucs,
        "vision_only": vision_only_aucs,
        "audio_60ms": audio_60ms_aucs,
        "audio_500ms": audio_500ms_aucs,
        "audio_1000ms": audio_1000ms_aucs,
        "baseline_mean": baseline_mean_aucs,
        "vision_slfm": vision_audio_slfm_1000ms_aucs,
        "vision_doa_360": vision_audio_doa_360_aucs,
        "vision_doa_360_slfm": vision_audio_slfm_doa_360_aucs
    }
   # keystep_annotations = keystep_json["annotations"]

    scenario_vids = json.load(open("scenario_vids.json", "r"))
    baseline_method = "vision_doa_360"
    our_method = "vision_doa_360_slfm"
    print("Comparing baseline method:", baseline_method, "with our method:", our_method)
    for scenario, vids in scenario_vids.items():
        scenario_auc_results = {}
        for method, per_video_aucs in auc_results.items():
            scenario_auc_results[method] = {vid: auc for vid, auc in per_video_aucs.items() if vid in vids}
        
        auc_k = ["auc@5", "auc@10", "auc@20"]
        auc_type =["total_aucs", "rotation_only_aucs", "translation_only_aucs"]
        avg_success_rate = 0
        for k in auc_k:
            for auc_t in auc_type:

                baseline_aucs = np.array([values[auc_t][k] for key, values in scenario_auc_results[baseline_method].items()])
                our_aucs = np.array([values[auc_t][k] for key, values in scenario_auc_results[our_method].items()])  # to avoid identical values
                # statistic, p_value = ks_2samp(baseline_aucs, our_aucs)

                # print(f"KS test for {auc_t} {k}: p-value: {p_value}")

                # Compare how many times our_aucs are better than baseline_aucs
                better_count = np.sum(our_aucs > baseline_aucs)
                total_count = len(baseline_aucs)
                avg_success_rate += better_count / total_count
                if "20" in k:
                    print(auc_t, k, f"Scenario: {scenario} | Our method {our_method} outperforms baseline {baseline_method} in {better_count/total_count*100:.2f}% of cases.")
                # print(f"Scenario: {scenario} | Our method outperforms baseline in {better_count} out of {total_count} videos.")
           
        avg_success_rate /= (len(auc_k) * len(auc_type))
        print(f"Scenario: {scenario} | Average Success Rate: {avg_success_rate:.4f}")
        print("-----")



    frame_pair_errs_vision_only = np.load("frame_pair_pose_errors_vision_only.npy", allow_pickle=True).item()
    frame_pair_errs_vision_audio_embed = np.load("frame_pair_pose_errors_audio_embed_vision_1000ms.npy", allow_pickle=True).item()
    frame_pair_errs_vision_doa_360 = np.load("frame_pair_pose_errors_doa_360_vision.npy", allow_pickle=True).item()
    frame_pair_errs_vision_slfm = np.load("frame_pair_pose_errors_slfm_embed.npy", allow_pickle=True).item()
    frame_pair_errs_vision_doa_360_slfm = np.load("frame_pair_pose_errors_slfm_and_doa_360.npy", allow_pickle=True).item()

    vid_to_scenario_map = {}
    for scenario in scenario_vids:
        for vid in scenario_vids[scenario]:
            vid_to_scenario_map[vid] = scenario

    methods = ["vision_only", "vision_audio_embed", "vision_doa_360", "vision_slfm", "vision_doa_360_slfm"]
    frame_pairs = {
        "vision_only": frame_pair_errs_vision_only,
        "vision_audio_embed": frame_pair_errs_vision_audio_embed,
        "vision_doa_360": frame_pair_errs_vision_doa_360,
        "vision_slfm": frame_pair_errs_vision_slfm,
        "vision_doa_360_slfm": frame_pair_errs_vision_doa_360_slfm,
    }

    # Fast batched zero-shot audio classification without pipeline
    classify_model = False
    if classify_model:
        model_name = "laion/clap-htsat-fused"
        try:
            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda:0") if use_cuda else torch.device("cpu")
            dtype = torch.float16 if use_cuda else torch.float32

            processor = AutoProcessor.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name, torch_dtype=dtype)
            model.to(device)
            model.eval()

            # Candidate labels (text) -> embed once
            candidate_labels_batch = [
                "human speaking in foreground",
                "human speaking in background",
                "silence",
                "playing music",
                "water flowing",
                "wind blowing",
                "ball bouncing",
                "breathing",
                "cooking food"
            ]
            with torch.no_grad():
                text_inputs = processor(text=candidate_labels_batch, padding=True, return_tensors="pt").to(device)
                text_features = model.get_text_features(**text_inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.detach()
            zero_shot_available = True
            target_sr = processor.feature_extractor.sampling_rate
        except Exception as e:
            zero_shot_available = False
            text_features = None
            target_sr = 16000  # fallback

        method = "vision_doa_360_slfm"  # choose method to batch over

        class FramePairDataset(Dataset):
            def __init__(self, frame_pair_dict, audio_root, duration_ms=1000):
                self.samples = []
                self.audio_root = audio_root
                self.duration_ms = duration_ms
                for frame_pair_info, errs in frame_pair_dict.items():
                    vid, source_frame, target_frame = frame_pair_info
                    self.samples.append({
                        "frame_pair_info": frame_pair_info,
                        "vid": vid,
                        "source_frame": source_frame,
                        "target_frame": target_frame,
                        "rotation_error": errs.get("rotation_error"),
                        "translation_error": errs.get("translation_error"),
                    })

            def _clip_path(self, vid, frame):
                return os.path.join(
                    audio_clip_dir,
                    vid,
                    "sound",
                    f"{frame:06d}_duration_{self.duration_ms}ms.wav"
                )

            def __getitem__(self, idx):
                s = self.samples[idx]
                return {
                    **s,
                    "source_path": self._clip_path(s["vid"], s["source_frame"]),
                    "target_path": self._clip_path(s["vid"], s["target_frame"]),
                }

            def __len__(self):
                return len(self.samples)

        def collate_fn(batch):
            return {
                "frame_pair_infos": [b["frame_pair_info"] for b in batch],
                "vids": [b["vid"] for b in batch],
                "source_frames": [b["source_frame"] for b in batch],
                "target_frames": [b["target_frame"] for b in batch],
                "source_paths": [b["source_path"] for b in batch],
                "target_paths": [b["target_path"] for b in batch],
                "rotation_errors": [b["rotation_error"] for b in batch],
                "translation_errors": [b["translation_error"] for b in batch],
            }

        dataset = FramePairDataset(frame_pairs[method], audio_clip_dir, duration_ms=1000)
        loader = DataLoader(dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)

        # Simple audio cache to avoid re-loading same file
        _audio_cache = {}

        def _load_audio(path):
            if path in _audio_cache:
                return _audio_cache[path]
            if not os.path.isfile(path):
                _audio_cache[path] = None
                return None
            try:
                wav, sr = sf.read(path)
            except Exception:
                try:
                    wav, sr = librosa.load(path, sr=None, mono=False)
                except Exception:
                    _audio_cache[path] = None
                    return None
            if wav is None:
                _audio_cache[path] = None
                return None
            if wav.ndim > 1:
                wav = np.mean(wav, axis=1)
            if sr != target_sr:
                try:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
                except Exception:
                    wav = resample_poly(wav, target_sr, sr)
                sr = target_sr
            wav = wav.astype(np.float32)
            _audio_cache[path] = wav
            return wav

        def _classify_batch(paths, labels_unused=None):
            if not zero_shot_available:
                return [{"label": "unknown", "score": 0.0} for _ in paths]

            waves = []
            valid_mask = []
            for p in paths:
                w = _load_audio(p)
                if w is None or len(w) == 0:
                    waves.append(np.zeros(int(target_sr * 1.0), dtype=np.float32))  # 1s zero pad
                    valid_mask.append(False)
                else:
                    waves.append(w)
                    valid_mask.append(True)

            try:
                with torch.no_grad():
                    inputs = processor(audios=waves, sampling_rate=target_sr, return_tensors="pt", padding=True)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    # Ensure dtype matches model weights (float16 on CUDA, float32 otherwise)
                    model_dtype = next(model.parameters()).dtype
                    inputs = {k: (v.to(model_dtype) if v.dtype.is_floating_point else v) for k, v in inputs.items()}
                    audio_features = model.get_audio_features(**inputs)
                    audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)
                    sims = audio_features @ text_features.T  # [B, L]
                    # Convert to probabilities (softmax over labels)
                    probs = torch.softmax(sims, dim=-1)
                    top_scores, top_indices = probs.max(dim=-1)
            except Exception as e:
                raise e
            #   return [{"label": "error", "score": 0.0} for _ in paths]

            preds = []
            for ok, score, idx in zip(valid_mask, top_scores.cpu().tolist(), top_indices.cpu().tolist()):
                if not ok:
                    preds.append({"label": "missing", "score": 0.0})
                else:
                    preds.append({"label": candidate_labels_batch[idx], "score": float(score)})
            return preds

        # JSONL output (truncate if exists)
        jsonl_path = "frame_pair_audio_preds.jsonl"
        with open(jsonl_path, "w") as _fw:
            pass

        with open(jsonl_path, "a") as fw:
            for batch in tqdm(loader, leave=False, desc="Classifying audio clips"):
                src_preds = _classify_batch(batch["source_paths"], candidate_labels_batch)
                tgt_preds = _classify_batch(batch["target_paths"], candidate_labels_batch)
                for fp, sp, tp in zip(batch["frame_pair_infos"], src_preds, tgt_preds):
                    entry = frame_pairs[method][fp]
                    entry["source_audio_label"] = sp["label"]
                    entry["source_audio_score"] = sp["score"]
                    entry["target_audio_label"] = tp["label"]
                    entry["target_audio_score"] = tp["score"]

                    # Build JSONL record: key=(fp, sp, tp), value=source label pair
                    record = {
                    
                        "frame_pair": list(fp),
                        # "source_pred": sp,
                        # "target_pred": tp,
                        "source_label": sp["label"],
                        "target_label": tp["label"],
                    }
                    print(record)
                    fw.write(json.dumps(record) + "\n")

    frame_pair_audio_preds_path = "frame_pair_audio_preds.jsonl"
    frame_pair_audio_preds = {}
    if os.path.isfile(frame_pair_audio_preds_path):
        with open(frame_pair_audio_preds_path, "r") as fr:
            for line in fr:
                record = json.loads(line)
                frame_pair_info = tuple(record["frame_pair"])
                frame_pair_audio_preds[frame_pair_info] = {
                    "source_label": record["source_label"],
                    "target_label": record["target_label"],
                }


    errs_per_scenario = {}
    errs_per_ambient_change = {"same": {},
                               "different": {}}
    # same_label_count = 0
    # total_count = 0
    same_label_count_per_scenario = {scenario: {"same": 0, "total": 0} for scenario in scenario_vids}
    for method in methods:
        for frame_pair_info, errs in tqdm(frame_pairs[method].items()):
            vid, source_frame, target_frame = frame_pair_info
            source_label = frame_pair_audio_preds[frame_pair_info]["source_label"]
            target_label = frame_pair_audio_preds[frame_pair_info]["target_label"]

            # same_label_count += int(source_label == target_label)
            # total_count += 1

            rotation_err = errs['rotation_error']
            translation_err = errs['translation_error']
            scenario = vid_to_scenario_map[vid]

            same_label_count_per_scenario[scenario]["same"] += int(source_label == target_label)
            same_label_count_per_scenario[scenario]["total"] += 1

        # print(f"Scenario: {scenario}, Video: {vid}, Rotation Error: {rotation_err}, Translation Error: {translation_err}")

            if scenario not in errs_per_scenario:
                errs_per_scenario[scenario] = {
                    "vision_only": {"rotation_errors": [], "translation_errors": []},
                    "vision_audio_embed": {"rotation_errors": [], "translation_errors": []},
                    "vision_doa_360": {"rotation_errors": [], "translation_errors": []},
                    "vision_slfm": {"rotation_errors": [], "translation_errors": []},
                    "vision_doa_360_slfm": {"rotation_errors": [], "translation_errors": []},

                }
            if method not in errs_per_ambient_change["same"]:
                errs_per_ambient_change["same"][method] = {"rotation_errors": [], "translation_errors": []}
            if method not in errs_per_ambient_change["different"]:
                errs_per_ambient_change["different"][method] = {"rotation_errors": [], "translation_errors": []}

            errs_per_scenario[scenario][method]["rotation_errors"].append(rotation_err)
            errs_per_scenario[scenario][method]["translation_errors"].append(translation_err)
            errs_per_ambient_change["same" if source_label == target_label else "different"][method]["rotation_errors"].append(rotation_err)
            errs_per_ambient_change["same" if source_label == target_label else "different"][method]["translation_errors"].append(translation_err)

    for scenario in same_label_count_per_scenario:
        same = same_label_count_per_scenario[scenario]["same"]
        total = same_label_count_per_scenario[scenario]["total"]
        rate = same / total if total > 0 else 0.0
        print(f"Scenario: {scenario} | Same audio label rate: {rate:.4f} ({same}/{total})")
    raise Exception("done")

    thresholds = [5,10,20]
    aucs_per_ambient_change = {"same": {}, "different": {}}
    rot_aucs_per_ambient_change = {"same": {}, "different": {}}
    trans_aucs_per_ambient_change = {"same": {}, "different": {}}
    for change_type in errs_per_ambient_change:
        for method in tqdm(errs_per_ambient_change[change_type], desc=f"Calculating AUCs for ambient change type {change_type}", leave=False):
            rotation_errors = np.array(errs_per_ambient_change[change_type][method]["rotation_errors"])
            translation_errors = np.array(errs_per_ambient_change[change_type][method]["translation_errors"])
            aucs = error_auc(rotation_errors, translation_errors, thresholds)
            aucs_per_ambient_change[change_type][method] = aucs
            rot_aucs_per_ambient_change[change_type][method] = error_auc(rotation_errors, np.zeros_like(rotation_errors), thresholds)
            trans_aucs_per_ambient_change[change_type][method] = error_auc(np.zeros_like(translation_errors), translation_errors, thresholds)
    pprint(aucs_per_ambient_change)
    os.makedirs("figs_auc_per_ambient_change", exist_ok=True)
    json.dump(aucs_per_ambient_change, open("figs_auc_per_ambient_change/aucs_per_ambient_change.json", "w"), indent=4)
    json.dump(rot_aucs_per_ambient_change, open("figs_auc_per_ambient_change/rot_aucs_per_ambient_change.json", "w"), indent=4)
    json.dump(trans_aucs_per_ambient_change, open("figs_auc_per_ambient_change/trans_aucs_per_ambient_change.json", "w"), indent=4)

    raise Exception("done")

    # calculate AUCs
    aucs_per_scenario = {}
    rot_aucs_per_scenario = {}
    trans_aucs_per_scenario = {}
    for scenario in scenario_vids:
        aucs_per_scenario[scenario] = {}
        rot_aucs_per_scenario[scenario] = {}
        trans_aucs_per_scenario[scenario] = {}
        for method in tqdm(errs_per_scenario[scenario], desc=f"Calculating AUCs for scenario {scenario}", leave=False):
            rotation_errors = np.array(errs_per_scenario[scenario][method]["rotation_errors"])
            translation_errors = np.array(errs_per_scenario[scenario][method]["translation_errors"])
            aucs = error_auc(rotation_errors, translation_errors, thresholds)
            aucs_per_scenario[scenario][method] = aucs
            rot_aucs_per_scenario[scenario][method] = error_auc(rotation_errors, np.zeros_like(rotation_errors), thresholds)
            trans_aucs_per_scenario[scenario][method] = error_auc(np.zeros_like(translation_errors), translation_errors, thresholds)
    
    pprint(aucs_per_scenario)
    json.dump(aucs_per_scenario, open("figs_auc_per_scenario/aucs_per_scenario.json", "w"), indent=4)
    json.dump(rot_aucs_per_scenario, open("figs_auc_per_scenario/rot_aucs_per_scenario.json", "w"), indent=4)
    json.dump(trans_aucs_per_scenario, open("figs_auc_per_scenario/trans_aucs_per_scenario.json", "w"), indent=4)

    # Plot AUCs per scenario (one figure per scenario)
    save_dir = "figs_auc_per_scenario"
    os.makedirs(save_dir, exist_ok=True)

    # Determine threshold keys order
    def _extract_threshold_keys(method_results):
        default = ["auc@5", "auc@10", "auc@20"]
        first = next(iter(method_results.values()))
        keys = list(first.keys())
        if set(default).issubset(keys):
            return default
        # Fallback: sort by numeric value in key
        def knum(k):
            m = re.findall(r"\d+", k)
            return int(m[0]) if m else float("inf")
        return sorted(keys, key=knum)

    for t, auc_data in enumerate([aucs_per_scenario, rot_aucs_per_scenario, trans_aucs_per_scenario]):
        for scenario, method_results in auc_data.items():
            if not method_results:
                continue

            thr_keys = _extract_threshold_keys(method_results)

            # Preserve a consistent method order if available
            try:
                plot_methods = [m for m in methods if m in method_results]
            except NameError:
                plot_methods = list(method_results.keys())

            x = np.arange(len(plot_methods))
            width = 0.22 if len(thr_keys) == 3 else max(0.1, 0.8 / max(1, len(thr_keys)))

            fig, ax = plt.subplots(figsize=(12, 6))
            all_y_vals = []
            for i, k in enumerate(thr_keys):
                y = [method_results[m].get(k, np.nan) for m in plot_methods]
                ax.bar(x + (i - (len(thr_keys) - 1) / 2) * width, y, width, label=k)
                all_y_vals.extend([v for v in y if np.isfinite(v)])

            ax.set_xticks(x)
            ax.set_xticklabels(plot_methods, rotation=20, ha="right")

            # Dynamic y-limit per scenario (a bit above max value)
            if all_y_vals:
                y_max = max(all_y_vals)
                top = y_max * 1.05 if y_max > 0 else 0.05
                ax.set_ylim(0, top)

            ax.set_ylabel("AUC")
            ax.set_title(f"AUC per method for scenario: {scenario}")
            ax.legend(title="Threshold")
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            fig.tight_layout()

            # safe_name = re.sub(r'[\\/*?:"<>| ]+', "_", str(scenario))
            plot_type = "total_auc" if t == 0 else ("rot_auc" if t == 1 else "trans_auc")
            fig.savefig(os.path.join(save_dir, f"{scenario}_{plot_type}.png"), dpi=200)
            plt.close(fig)



    # auc_k = ["auc@5", "auc@10", "auc@20"]
    # auc_type =["total_aucs", "rotation_only_aucs", "translation_only_aucs"]
    # for k in auc_k:
    #     for auc_t in auc_type:
    #         baseline_aucs = np.array([values[auc_t][k] for key, values in auc_results["vision_only"].items()])
    #         our_aucs = np.array([values[auc_t][k] for key, values in auc_results["vision_doa_360_slfm"].items()])  # to avoid identical values
    #         statistic, p_value = ks_2samp(baseline_aucs, our_aucs)

    #         print(f"KS test for {auc_t} {k}: p-value: {p_value}")

    #         # Compare how many times our_aucs are better than baseline_aucs
    #         better_count = np.sum(our_aucs > baseline_aucs)
    #         total_count = len(baseline_aucs)
    #         print(f"Our method outperforms baseline in {better_count} out of {total_count} videos.")

    # baseline_aucs = np.array([values["total_aucs"]["auc@20"] for key, values in auc_results["vision_audio_doa_1000ms"].items()])
    # our_aucs = np.array([values["total_aucs"]["auc@20"] for key, values in auc_results["vision_slfm_1000ms"].items()])  # to avoid identical values

    # statistic, p_value = ks_2samp(baseline_aucs, our_aucs)

    # print(f"KS test statistic: {statistic}, p-value: {p_value}")