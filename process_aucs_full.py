import json
from pprint import pprint
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from tqdm import tqdm

keystep_json = json.load(open("/vision/vision_data_2/egoexo4d_audio/annotations/keystep_val.json", "r"))
takes_json = json.load(open("/vision/vision_data_2/EgoExo4D_public_v1/takes.json", "r"))

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
        "vision_slfm_1000ms": vision_audio_slfm_1000ms_aucs
    }
   # keystep_annotations = keystep_json["annotations"]

    # TODO

    auc_results_by_scenario = {method: {} for method in auc_results.keys()}
    scenario_vid_counts = {}
    scenario_vids = {}
    for take in takes_json:
        take_name = take['take_name']
        scenario = take["parent_task_name"]
        for method, results in auc_results.items():
            if take_name in results:
                print(f"Take: {take_name}")

                
            #    auc_results_by_scenario[method][scenario] = [results[take_name]]
                if scenario not in auc_results_by_scenario[method]:
                    auc_results_by_scenario[method][scenario] = []
                auc_results_by_scenario[method][scenario].append(results[take_name])
                print(f"{method} AUCs for {take_name}:")
                pprint(results[take_name])

                if scenario not in scenario_vids:
                    scenario_vids[scenario] = set()
                scenario_vids[scenario] |= {take_name}
            # else:
            #     print(f"No AUCs found for {take_name} in this result set.")
        #scenario_vid_counts[scenario] = len(auc_results_by_scenario[method][scenario])
           
        print("\n") 


    with open('scenario_vids.json', 'w') as f:
        json.dump({k: list(v) for k, v in scenario_vids.items()}, f, indent=2)

    avgs_by_scenario = {}
    for method, scenarios in auc_results_by_scenario.items():
        print(f"Method: {method}")
        avgs_by_scenario[method] = {}
        for scenario, results in scenarios.items():
            print(f"Scenario: {scenario}, Number of Takes: {len(results)}")
            # calculate average AUCs for each scenario
           # print(results)
            # based on how auc_results_by_scenario is structured, calculate the average AUCs for each scenario (AUC@5, AUC@10, and AUC@20)
            # if results:
            #     avg_aucs = np.mean([list(res.values()) for res in results], axis=0)
            #     print(f"Average AUCs for {scenario}: {avg_aucs}")
            # else:
            #     print(f"No results for scenario {scenario}")
            avgs = {}
            for auc_info in results:
                for auc_name, auc_values in auc_info.items():
                    for auc_acc_label, auc_accuracy in auc_values.items():
                        if auc_name not in avgs:
                            avgs[auc_name] = {auc_acc_label: [] for auc_acc_label in auc_values.keys()}
                        avgs[auc_name][auc_acc_label].append(auc_accuracy)
                    # replace lists with average values
            for auc_name, auc_values in avgs.items():
                for auc_acc_label, auc_accuracy in auc_values.items():
                    avgs[auc_name][auc_acc_label] = np.mean(auc_accuracy)
            print(f"Average AUCs for {scenario}:")
            # print(avgs)
            for auc_name, auc_values in avgs.items():
                print(f"{auc_name}: {auc_values}")

            #print(avgs)
            print("\n")
            avgs_by_scenario[method][scenario] = avgs

    pprint(avgs_by_scenario)

    auc_results_tupled = {}
    for method, scenarios in avgs_by_scenario.items():
        for scenario, results in scenarios.items():
            if scenario not in auc_results_tupled:
                auc_results_tupled[scenario] = {}
            
            for auc_name, auc_values in results.items():
                if auc_name not in auc_results_tupled[scenario]:
                    auc_results_tupled[scenario][auc_name] = {}
                
                for auc_acc_label, auc_accuracy in auc_values.items():
                    if auc_acc_label not in auc_results_tupled[scenario][auc_name]:
                        auc_results_tupled[scenario][auc_name][auc_acc_label] = {}
                    auc_results_tupled[scenario][auc_name][auc_acc_label][method] = auc_accuracy

    # save auc_results_tupled to json
    with open('auc_results_by_scenario.json', 'w') as f:
        json.dump(auc_results_tupled, f, indent=2)

    # count_num_maxes = {}
    # max_counts = {}
    # for scenario, auc_data in auc_results_tupled.items():
    # Plot auc_results_tupled as bar charts: one figure per auc@k (auc_acc_label)
    # Structure: auc_results_tupled[scenario][auc_name][auc_acc_label][method] = value

    def _sanitize(s: str) -> str:
        return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in s)

    # Collect keys
    scenarios = sorted(auc_results_tupled.keys())
    auc_names = sorted({auc_name for sc in scenarios for auc_name in auc_results_tupled[sc].keys()})
    auc_acc_labels = sorted({
        auc_acc_label
        for sc in scenarios
        for auc_name in auc_results_tupled[sc].keys()
        for auc_acc_label in auc_results_tupled[sc][auc_name].keys()
    })
    methods = sorted({
        method
        for sc in scenarios
        for auc_name in auc_results_tupled[sc].keys()
        for auc_acc_label in auc_results_tupled[sc][auc_name].keys()
        for method in auc_results_tupled[sc][auc_name][auc_acc_label].keys()
    })

    os.makedirs("full_egoexo4d_plots", exist_ok=True)

    # import matplotlib.pyplot as plt

    # # For each auc_name, we will create one figure per auc_acc_label (e.g., auc@5)
    # for auc_name in auc_names:
    #     # Gather labels specific to this auc_name
    #     acc_labels_for_auc_name = sorted({
    #         lbl for sc in scenarios
    #         for lbl in auc_results_tupled[sc].get(auc_name, {}).keys()
    #     })
    #     for auc_acc_label in acc_labels_for_auc_name:
    #         # Build data matrix: rows = scenarios, cols = methods
    #         data = []
    #         for sc in scenarios:
    #             row = []
    #             methods_dict = auc_results_tupled.get(sc, {}).get(auc_name, {}).get(auc_acc_label, {})
    #             for m in methods:
    #                 row.append(methods_dict.get(m, np.nan))
    #             data.append(row)
    #         data = np.array(data, dtype=float)

    #         n_groups = len(scenarios)
    #         n_methods = len(methods)
    #         if n_groups == 0 or n_methods == 0:
    #             continue

    #         # Determine y-axis max for this graph
    #         if np.all(np.isnan(data)):
    #             y_max = 1.0
    #         else:
    #             y_max = float(np.nanmax(data)) 

    #         y_max *= 1.05

    #         x = np.arange(n_groups)
    #         group_width = 0.8
    #         bar_width = group_width / n_methods
    #         offsets = (np.arange(n_methods) - (n_methods - 1) / 2.0) * bar_width

    #         fig_w = max(10, int(0.5 * n_groups * max(1, n_methods / 4)))
    #         plt.figure(figsize=(fig_w, 6))
    #         for i, m in enumerate(methods):
    #             plt.bar(x + offsets[i], data[:, i], width=bar_width, label=m)

    #         plt.title(f"{auc_name} - {auc_acc_label}")
    #         plt.ylabel("AUC")
    #         plt.ylim(0, y_max)
    #         plt.xticks(x, scenarios, rotation=45, ha="right")
    #         plt.legend(
    #             loc="upper center",
    #             bbox_to_anchor=(0.5, -0.15),
    #             ncol=min(4, n_methods),
    #             fontsize="small",
    #             borderaxespad=0,
    #         )
    #         plt.tight_layout()

    #         fname = f"full_egoexo4d_plots/{_sanitize(auc_name)}__{_sanitize(auc_acc_label)}.png"
    #      #   plt.savefig(fname, dpi=200)
    #         plt.close()
    #         print(f"Saved {fname}")

    baseline_mean_angle_range_aucs = json.load(open("angle_range_auc_results_baseline_mean.json", "r"))
    vision_only_angle_range_aucs = json.load(open("angle_range_auc_results_vision_only.json", "r"))
    audio_60ms_angle_range_aucs = json.load(open("angle_range_auc_results_audio_only_60ms.json", "r"))
    audio_500ms_angle_range_aucs = json.load(open("angle_range_auc_results_audio_only_500ms.json", "r"))
    audio_1000ms_angle_range_aucs = json.load(open("angle_range_auc_results_audio_only_1000ms.json", "r"))
    vision_audio_60ms_angle_range_aucs = json.load(open("angle_range_auc_results_audio_embed_60ms.json", "r"))
    vision_audio_500ms_angle_range_aucs = json.load(open("angle_range_auc_results_audio_embed_500ms.json", "r"))
    vision_audio_1000ms_angle_range_aucs = json.load (open("angle_range_auc_results_audio_embed_1000ms.json", "r"))
    audio_doa_60ms_angle_range_aucs = json.load(open("angle_range_auc_results_doa_cp_60ms.json", "r"))
    audio_doa_500ms_angle_range_aucs = json.load(open("angle_range_auc_results_doa_cp_500ms.json", "r"))
    audio_doa_1000ms_angle_range_aucs = json.load(open("angle_range_auc_results_doa_cp_1000ms.json", "r"))
    vision_audio_doa_60ms_angle_range_aucs = json.load(open("angle_range_auc_results_doa_embed_vision_60ms.json", "r"))
    vision_audio_doa_500ms_angle_range_aucs = json.load(open("angle_range_auc_results_doa_embed_vision_500ms.json", "r"))
    vision_audio_doa_1000ms_angle_range_aucs = json.load(open("angle_range_auc_results_doa_embed_vision_1000ms.json", "r"))
    angle_range_auc_results = {
        "audio_doa_60ms": audio_doa_60ms_angle_range_aucs,
        "audio_doa_500ms": audio_doa_500ms_angle_range_aucs,
        "audio_doa_1000ms": audio_doa_1000ms_angle_range_aucs,
        "vision_audio_doa_60ms": vision_audio_doa_60ms_angle_range_aucs,
        "vision_audio_doa_500ms": vision_audio_doa_500ms_angle_range_aucs,
        "vision_audio_doa_1000ms": vision_audio_doa_1000ms_angle_range_aucs,
        "vision_audio_60ms": vision_audio_60ms_angle_range_aucs,
        "vision_audio_500ms": vision_audio_500ms_angle_range_aucs,
        "vision_audio_1000ms": vision_audio_1000ms_angle_range_aucs,
        "vision_only": vision_only_angle_range_aucs,
        "audio_60ms": audio_60ms_angle_range_aucs,
        "audio_500ms": audio_500ms_angle_range_aucs,
        "audio_1000ms": audio_1000ms_angle_range_aucs,
        "baseline_mean": baseline_mean_angle_range_aucs
    }

    # Aggregate angle-range AUCs across methods and plot by AUC score and angle range

    def _is_metrics_leaf(d):
        # Expect: {auc_name: {auc@k: float, ...}, ...}
        if not isinstance(d, dict):
            return False
        for v in d.values():
            if isinstance(v, dict) and all(isinstance(x, (int, float)) for x in v.values()):
                return True
        return False

    def _aggregate_angle_results(results_dict):
        # Returns: angle_range -> auc_name -> auc_label -> mean_value
        accum = {}
        def _add(angle_label, metrics):
            angle_label = str(angle_label)
            for auc_name, auc_vals in metrics.items():
                if not isinstance(auc_vals, dict):
                    continue
                for auc_label, val in auc_vals.items():
                    if not isinstance(val, (int, float)):
                        continue
                    accum.setdefault(angle_label, {}).setdefault(auc_name, {}).setdefault(auc_label, []).append(float(val))

        if isinstance(results_dict, dict):
            for k, v in results_dict.items():
                if _is_metrics_leaf(v):
                    _add(k, v)
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        if _is_metrics_leaf(vv):
                            _add(kk, vv)

        out = {}
        for angle_label, aucs in accum.items():
            out[angle_label] = {}
            for auc_name, labels in aucs.items():
                out[angle_label][auc_name] = {lbl: float(np.mean(vals)) for lbl, vals in labels.items() if vals}
        return out

    # Build tuple: angle_range -> auc_name -> auc_label -> method -> value
    angle_tupled = {}
    for method, res in angle_range_auc_results.items():
        agg = _aggregate_angle_results(res)
        for angle_label, aucs in agg.items():
            angle_tupled.setdefault(angle_label, {})
            for auc_name, labels in aucs.items():
                angle_tupled[angle_label].setdefault(auc_name, {})
                for auc_label, val in labels.items():
                    angle_tupled[angle_label][auc_name].setdefault(auc_label, {})
                    angle_tupled[angle_label][auc_name][auc_label][method] = val

    # Persist aggregated results
    with open('angle_range_auc_results_aggregated.json', 'w') as f:
        json.dump(angle_tupled, f, indent=2)

    # Sorting helper for angle range labels
    def _angle_sort_key(s: str):
        nums = re.findall(r"[-+]?\d*\.?\d+", s)
        lower = float(nums[0]) if nums else float('inf')
        return (lower, s)

    # Collect keys
    angle_ranges = sorted(angle_tupled.keys(), key=_angle_sort_key)
    auc_names = sorted({auc_name for ang in angle_ranges for auc_name in angle_tupled[ang].keys()})
    auc_labels = sorted({
        lbl for ang in angle_ranges for auc_name in angle_tupled[ang].keys() for lbl in angle_tupled[ang][auc_name].keys()
    })
    methods = sorted({
        m for ang in angle_ranges for auc_name in angle_tupled[ang].keys()
        for lbl in angle_tupled[ang][auc_name].keys()
        for m in angle_tupled[ang][auc_name][lbl].keys()
    })

    os.makedirs("angle_range_plots", exist_ok=True)

    # for auc_name in auc_names:
    #     labels_for_auc = sorted({lbl for ang in angle_ranges for lbl in angle_tupled[ang].get(auc_name, {}).keys()})
    #     for auc_label in labels_for_auc:
    #         # Build data matrix: rows = angle ranges, cols = methods
    #         data = []
    #         for ang in angle_ranges:
    #             md = angle_tupled.get(ang, {}).get(auc_name, {}).get(auc_label, {})
    #             data.append([md.get(m, np.nan) for m in methods])
    #         data = np.array(data, dtype=float)

    #         n_groups = len(angle_ranges)
    #         n_methods = len(methods)
    #         if n_groups == 0 or n_methods == 0:
    #             continue

    #         y_max = 1.0 if np.all(np.isnan(data)) else float(np.nanmax(data))
    #         y_max *= 1.05

    #         x = np.arange(n_groups)
    #         group_width = 0.8
    #         bar_width = group_width / n_methods
    #         offsets = (np.arange(n_methods) - (n_methods - 1) / 2.0) * bar_width

    #         fig_w = max(10, int(0.4 * n_groups * max(1, n_methods / 4)))
    #         plt.figure(figsize=(fig_w, 6))
    #         for i, m in enumerate(methods):
    #             plt.bar(x + offsets[i], data[:, i], width=bar_width, label=m)

    #         plt.title(f"{auc_name} - {auc_label} by angle range")
    #         plt.ylabel("AUC")
    #         plt.ylim(0, y_max)
    #         plt.xticks(x, angle_ranges, rotation=45, ha="right")
    #         plt.legend(
    #             loc="upper center",
    #             bbox_to_anchor=(0.5, -0.15),
    #             ncol=min(4, n_methods),
    #             fontsize="small",
    #             borderaxespad=0,
    #         )
    #         plt.tight_layout()
    #         fname = f"angle_range_plots/{_sanitize(auc_name)}__{_sanitize(auc_label)}.png"
    #         plt.savefig(fname, dpi=200)
    #         plt.close()
    #         print(f"Saved {fname}")


    frame_pairs_vision_only = np.load("frame_pair_pose_errors_vision_only.npy", allow_pickle=True).item()
    frame_pairs_vision_audio_embed_60ms = np.load("frame_pair_pose_errors_audio_embed_vision_60ms.npy", allow_pickle=True).item()
    frame_pairs_vision_audio_embed_1000ms = np.load("frame_pair_pose_errors_audio_embed_vision_1000ms.npy", allow_pickle=True).item()
    frame_pairs_doa_embed_vision_60ms = np.load("frame_pair_pose_errors_doa_embed_vision_60ms.npy", allow_pickle=True).item()
    frame_pairs_doa_embed_vision_500ms = np.load("frame_pair_pose_errors_doa_embed_vision_500ms.npy", allow_pickle=True).item()
    frame_pairs_doa_embed_vision_1000ms = np.load("frame_pair_pose_errors_doa_embed_vision_1000ms.npy", allow_pickle=True).item()
    frame_pairs_audio_only_60ms = np.load("frame_pair_pose_errors_audio_only_60ms.npy", allow_pickle=True).item()
    frame_pairs_audio_only_500ms = np.load("frame_pair_pose_errors_audio_only_500ms.npy", allow_pickle=True).item()
    frame_pairs_audio_only_1000ms = np.load("frame_pair_pose_errors_audio_only_1000ms.npy", allow_pickle=True).item()
    frame_pairs_doa_cp_60ms = np.load("frame_pair_pose_errors_doa_cp_60ms.npy", allow_pickle=True).item()
    frame_pairs_doa_cp_500ms = np.load("frame_pair_pose_errors_doa_cp_500ms.npy", allow_pickle=True).item()
    frame_pairs_doa_cp_1000ms = np.load("frame_pair_pose_errors_doa_cp_1000ms.npy", allow_pickle=True).item()
    frame_pairs_baseline_mean = np.load("frame_pair_pose_errors_baseline_mean.npy", allow_pickle=True).item()

   # pprint(frame_pairs_vision_only)
    
    frame_pair_results = {
        "vision_only": frame_pairs_vision_only,
        "vision_audio_embed_60ms": frame_pairs_vision_audio_embed_60ms,
        "vision_audio_embed_1000ms": frame_pairs_vision_audio_embed_1000ms,
        "doa_embed_vision_60ms": frame_pairs_doa_embed_vision_60ms,
        "doa_embed_vision_500ms": frame_pairs_doa_embed_vision_500ms,
        "doa_embed_vision_1000ms": frame_pairs_doa_embed_vision_1000ms,
        "audio_only_60ms": frame_pairs_audio_only_60ms,
        "audio_only_500ms": frame_pairs_audio_only_500ms,
        "audio_only_1000ms": frame_pairs_audio_only_1000ms,
        "doa_cp_60ms": frame_pairs_doa_cp_60ms,
        "doa_cp_500ms": frame_pairs_doa_cp_500ms,
        "doa_cp_1000ms": frame_pairs_doa_cp_1000ms,
        "baseline_mean": frame_pairs_baseline_mean
    }

    lowest_rotation_error_method_counts = {}
    lowest_translation_error_method_counts = {}
    lowest_total_error_method_counts = {}

    min_err_tracker = {}
    mean_err_tracker = {"rotation":{}, "translation": {}, "total":{}, "gt_rot": {}, "gt_trans_mag": {}}

    scenario_vids = json.load(open("scenario_vids.json", "r"))
    scenario_list = list(scenario_vids.keys())

    vid_name_to_scenario_map = {}
    for scenario, vids in scenario_vids.items():
        for vid in vids:
            vid_name_to_scenario_map[vid] = scenario

    filter_on = True
    ground_truths = json.load(open("ground_truth_pose_diffs.json", "r"))
    ground_truths = {eval(k): v for k,v in ground_truths.items()}
    for method, results in tqdm(frame_pair_results.items()):
       # print(results, type(results))
        if "only_1000ms" in method or "vision_only" in method or not filter_on: # "vision_only"
            current_min_method = None
            for frame_pair, data in results.items():
                lowest_rotation_error_method_counts[method] = lowest_rotation_error_method_counts.get(method, 0)
                lowest_translation_error_method_counts[method] = lowest_translation_error_method_counts.get(method, 0)
                lowest_total_error_method_counts[method] = lowest_total_error_method_counts.get(method, 0)

                min_err_tracker[frame_pair] = min_err_tracker.get(frame_pair, {
                    "min_rotation_error": float('inf'),
                    "min_translation_error": float('inf'),
                    "min_total_error": float('inf'),
                    "best_rotation_method": None,
                    "best_translation_method": None,
                    "best_total_method": None,
                    "full_rerr_map": {},
                    "full_terr_map": {},
                    'full_total_err_map': {},
                    "gt_rot": ground_truths[frame_pair]["gt_rot"],
                    "gt_trans_mag": ground_truths[frame_pair]["gt_trans_mag"]
                })

                rerr = float(data['rotation_error'])
                terr = float(data['translation_error'])
                total_err = max(rerr, terr)

                if rerr < min_err_tracker[frame_pair]["min_rotation_error"]:
                    min_err_tracker[frame_pair]["min_rotation_error"] = float( rerr)
                    min_err_tracker[frame_pair]["best_rotation_method"] = method
                if terr < min_err_tracker[frame_pair]["min_translation_error"]:
                    min_err_tracker[frame_pair]["min_translation_error"] = float(terr)
                    min_err_tracker[frame_pair]["best_translation_method"] = method               
                if total_err < min_err_tracker[frame_pair]["min_total_error"]:
                    min_err_tracker[frame_pair]["min_total_error"] = float(total_err)
                    min_err_tracker[frame_pair]["best_total_method"] = method

                min_err_tracker[frame_pair]['full_rerr_map'][method] = rerr
                min_err_tracker[frame_pair]['full_terr_map'][method] = terr
                min_err_tracker[frame_pair]['full_total_err_map'][method] = total_err

            
   ## pprint(min_err_tracker)
     
    # save min_err_tracker to json
    # with open('min_err_tracker_naive_audio_filtered.json', 'w') as f:
    #     json.dump({str(k) : v for k, v in min_err_tracker.items()}, f, indent=2)
    
    lowest_error_counts_by_scenario = { scenario : {} for scenario in scenario_list}
    # lowest_error_counts_by_scenario[scenario][method] = count of times method had lowest total error in that scenario

    for frame_pair in min_err_tracker:
        lowest_rotation_method = min_err_tracker[frame_pair]["best_rotation_method"]
        lowest_translation_method = min_err_tracker[frame_pair]["best_translation_method"]
        lowest_total_method = min_err_tracker[frame_pair]["best_total_method"]

        lowest_rotation_error_method_counts[lowest_rotation_method] = lowest_rotation_error_method_counts.get(lowest_rotation_method, 0) + 1
        lowest_translation_error_method_counts[lowest_translation_method] = lowest_translation_error_method_counts.get(lowest_translation_method, 0) + 1
        lowest_total_error_method_counts[lowest_total_method] = lowest_total_error_method_counts.get(lowest_total_method, 0) + 1

        mean_err_tracker["rotation"][lowest_rotation_method] = mean_err_tracker["rotation"].get(lowest_rotation_method, 0) + min_err_tracker[frame_pair]["min_rotation_error"]
        mean_err_tracker["translation"][lowest_translation_method] = mean_err_tracker["translation"].get(lowest_translation_method, 0) + min_err_tracker[frame_pair]["min_translation_error"]
        mean_err_tracker["total"][lowest_total_method] = mean_err_tracker["total"].get(lowest_total_method, 0) + min_err_tracker[frame_pair]["min_total_error"]

        n = lowest_total_error_method_counts[lowest_total_method] 

        lowest_error_counts_by_scenario[vid_name_to_scenario_map[frame_pair[0]]][lowest_total_method] = lowest_error_counts_by_scenario[vid_name_to_scenario_map[frame_pair[0]]].get(lowest_total_method, 0) + 1

        # CHANGE THIS TO LIST
        # mean_err_tracker["gt_rot"][lowest_total_method] = mean_err_tracker["gt_rot"].get(lowest_total_method, 0) * (n-1)/n + min_err_tracker[frame_pair]["gt_rot"]/n
        # mean_err_tracker["gt_trans_mag"][lowest_total_method] = mean_err_tracker["gt_trans_mag"].get(lowest_total_method, 0) * (n-1)/n + min_err_tracker[frame_pair]["gt_trans_mag"]/n
       
        if lowest_total_method not in mean_err_tracker["gt_rot"]:
            mean_err_tracker["gt_rot"][lowest_total_method] = []
        if lowest_total_method not in mean_err_tracker["gt_trans_mag"]:
            mean_err_tracker["gt_trans_mag"][lowest_total_method] = []

        mean_err_tracker["gt_rot"][lowest_total_method].append(min_err_tracker[frame_pair]["gt_rot"])
        mean_err_tracker["gt_trans_mag"][lowest_total_method].append(min_err_tracker[frame_pair]["gt_trans_mag"])


    # raise Exception(lowest_error_counts_by_scenario)

    # pprint(mean_err_tracker["gt_rot"])
    # pprint(mean_err_tracker["gt_trans_mag"])
    # raise Exception("done")

    for method in mean_err_tracker["gt_rot"]:
        median = float(np.median(mean_err_tracker["gt_rot"][method]))
        mean = float(np.mean(mean_err_tracker["gt_rot"][method]))
        ptl_25 = float(np.percentile(mean_err_tracker["gt_rot"][method], 25))
        ptl_75 = float(np.percentile(mean_err_tracker["gt_rot"][method], 75))
        ptl_95 = float(np.percentile(mean_err_tracker["gt_rot"][method], 95))
        std = float(np.std(mean_err_tracker["gt_rot"][method]))
        print(f"Method: {method}, GT Rot Median: {median}, Mean: {mean}, 25th Percentile: {ptl_25}, 75th Percentile: {ptl_75}, Std: {std}")
        print("Max", np.max(mean_err_tracker["gt_rot"][method]), "Min", np.min(mean_err_tracker["gt_rot"][method]), "95th Percentile", ptl_95)
    print()
    for method in mean_err_tracker["gt_trans_mag"]:
        median = float(np.median(mean_err_tracker["gt_trans_mag"][method]))
        mean = float(np.mean(mean_err_tracker["gt_trans_mag"][method]))
        ptl_25 = float(np.percentile(mean_err_tracker["gt_trans_mag"][method], 25))
        ptl_75 = float(np.percentile(mean_err_tracker["gt_trans_mag"][method], 75))
        std = float(np.std(mean_err_tracker["gt_trans_mag"][method]))
    
        print(f"Method: {method}, GT Trans Mag Median: {median}, Mean: {mean}, 25th Percentile: {ptl_25}, 75th Percentile: {ptl_75}, Std: {std}")
        print("Max", np.max(mean_err_tracker["gt_trans_mag"][method]), "Min", np.min(mean_err_tracker["gt_trans_mag"][method]))
    print()

    #raise Exception("done")
    pprint("Lowest Rotation Error Method Counts:")
    pprint(lowest_rotation_error_method_counts)
    pprint("Lowest Translation Error Method Counts:")
    pprint(lowest_translation_error_method_counts)
    pprint("Lowest Total Error Method Counts:")
    pprint(lowest_total_error_method_counts)

    # print("Mean Errors for Best Method Instances:")
    # for err_type in mean_err_tracker:
    #     print(f"{err_type.capitalize()} Error:")
    #     total_counts = sum(lowest_rotation_error_method_counts.values()) if err_type == "rotation" else \
    #                    sum(lowest_translation_error_method_counts.values()) if err_type == "translation" else \
    #                    sum(lowest_total_error_method_counts.values())
    #     for method, total_err in mean_err_tracker[err_type].items():
    #         count = lowest_rotation_error_method_counts[method] if err_type == "rotation" else \
    #                 lowest_translation_error_method_counts[method] if err_type == "translation" else \
    #                 lowest_total_error_method_counts[method]
    #         mean_err = total_err / count if count > 0 else float('inf')
    #         proportion = count / total_counts if total_counts > 0 else 0
    #         print(f"  {method}: Mean {err_type} error = {mean_err:.4f}, Count = {count}, Proportion = {proportion:.4f}")
    #     print()