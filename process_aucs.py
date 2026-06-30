import json
from pprint import pprint
import numpy as np
import os
import matplotlib.pyplot as plt

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
    audio_embed_aucs = json.load(open("per_video_auc_results_audio_embed_doa.json", "r"))
   # gradient_blending_aucs = json.load(open("per_video_auc_results_gb-va.json", "r"))

    # audio_embed_corrupted_aucs = json.load(open("per_video_auc_results_audio_embed_corrupted_vision.json", "r"))
    # corrupted_aucs = json.load(open("per_video_auc_results_corrupted_vision.json", "r"))
    # audio_embed_corrupted_aucs = json.load(open("per_video_auc_results_corrupted_vision+audio_test_only.json", "r"))
    # corrupted_aucs = json.load(open("per_video_auc_results_corrupted_vision_test_only.json", "r"))
    video_aucs = json.load(open("per_video_auc_results.json", "r"))
    audio_only_aucs = json.load(open("per_video_auc_results_audio_only.json"))
    baseline_mean_aucs = json.load(open("per_video_auc_results_baseline_mean.json", "r"))
    auc_results = {
        "audio_doa_60ms": ...,
        "audio_doa_500ms": ...,
        "audio_doa_1000ms": ...,
        "vision_audio_doa_60ms": ...,
        "vision_audio_doa_500ms": ...,
        "vision_audio_doa_1000ms": ...,
        "vision_audio_60ms": ...,
        "vision_audio_500ms": ...,
        "vision_audio_1000ms": ...,
        "vision_only": video_aucs,
        "audio_60ms": audio_only_aucs,
        "audio_500ms": ...,
        "audio_1000ms": ...,
        "baseline_mean": baseline_mean_aucs

    }
   # keystep_annotations = keystep_json["annotations"]

    auc_results_by_scenario = {"audio_embed_doa": {}, "gradient_blending": {}, "video": {}, "audio_only": {}, "baseline_mean": {}}
    scenario_vid_counts = {}
    scenario_vids = {}
    for take in takes_json.keys():
        take_name = take['take_name']
        for method, results in auc_results.items():
            if take_name in results:
                print(f"Take: {take_name}")

                scenario = take["parent_task_name"]
            #    auc_results_by_scenario[method][scenario] = [results[take_name]]
                if scenario not in auc_results_by_scenario[method]:
                    auc_results_by_scenario[method][scenario] = []
                auc_results_by_scenario[method][scenario].append(results[take_name])
                print(f"{method} AUCs for {take_name}:")
                pprint(results[take_name])

                if scenario not in scenario_vids:
                    scenario_vids[scenario] = set()
                scenario_vids[scenario] |= {take_name}
           #     auc_results_by_scenario[method][scenario]
            # else:
            #     print(f"No AUCs found for {take_name} in this result set.")
            scenario_vid_counts[scenario] = len(auc_results_by_scenario[method][scenario])
           
        print("\n")
    # for take_uid in keystep_annotations.keys():
    #     take_name = keystep_annotations[take_uid]["take_name"]
    #     for method, results in auc_results.items():
    #         if take_name in results:
    #             print(f"Take: {take_name}")

    #             scenario = keystep_annotations[take_uid]["scenario"]
    #         #    auc_results_by_scenario[method][scenario] = [results[take_name]]
    #             if scenario not in auc_results_by_scenario[method]:
    #                 auc_results_by_scenario[method][scenario] = []
    #             auc_results_by_scenario[method][scenario].append(results[take_name])
    #             print(f"{method} AUCs for {take_name}:")
    #             pprint(results[take_name])

    #             if scenario not in scenario_vids:
    #                 scenario_vids[scenario] = set()
    #             scenario_vids[scenario] |= {take_name}
    #             auc_results_by_scenario[method][scenario]
    #         # else:
    #         #     print(f"No AUCs found for {take_name} in this result set.")
    #         scenario_vid_counts[scenario] = len(auc_results_by_scenario[method][scenario])
           
    #     print("\n")
    # print lengths for each auc
    pprint(scenario_vid_counts)
    pprint(scenario_vids)
    with open('scenario_vids.json', 'w') as f:
        json.dump({k: list(v) for k, v in scenario_vids.items()}, f, indent=2)
   # exit(0)
   # print lengths of each scenario
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
    # # reorganize the results to have auc results tupled by method
    # auc_results_by_method = {}
    # for method, scenarios in avgs_by_scenario.items():
    #     for scenario, results in scenarios.items():
    #         if method not in auc_results_by_method:
    #             auc_results_by_method[method] = {}
    #         auc_results_by_method[method][scenario] = results
    # Reorganize to group AUC results by method as tuples
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

    # Find the method with maximum AUC for each group
    count_num_maxes = {}
    max_counts = {}
    for scenario, auc_data in auc_results_tupled.items():
        for auc_name, auc_values in auc_data.items():
            for auc_acc_label, methods in auc_values.items():
                max_method = max(methods.items(), key=lambda x: x[1])
                auc_results_tupled[scenario][auc_name][auc_acc_label]["_best_method"]= {
                    'method': max_method[0],
                    'value': max_method[1]
                }
                if scenario not in count_num_maxes:
                    count_num_maxes[scenario] = {}
                count_num_maxes[scenario][max_method[0]] = count_num_maxes[scenario].get(max_method[0], 0) + 1
        auc_results_tupled[scenario]["_num_maxes"] = count_num_maxes[scenario]
        max_counts[scenario] = count_num_maxes[scenario]
    # Print the best performing methods for each scenario and AUC metric
    print("Best performing methods for each scenario and AUC metric:")

    pprint(auc_results_tupled)
    pprint(auc_results_tupled.keys())
    print(len(auc_results_tupled.keys()))
    # pprint(max_counts)

    # import matplotlib.pyplot as plt

    # # Create plots for each scenario
    for scenario in auc_results_tupled.keys():
        if scenario.startswith('_'):  # Skip metadata keys
            continue
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'AUC Results for {scenario}', fontsize=16)
        
        # Get AUC names and accuracy labels
        auc_names = [k for k in auc_results_tupled[scenario].keys() if not k.startswith('_')]
        
        for i, auc_name in enumerate(auc_names):
            if i >= 4:  # Only plot first 4 AUC types
                break
                
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            # Get accuracy labels for this AUC
            acc_labels = [k for k in auc_results_tupled[scenario][auc_name].keys() if not k.startswith('_')]
            
            methods = ['audio_embed_doa', 'gradient_blending', 'video', "audio_only", 'baseline_mean']
            method_values = {method: [] for method in methods}
            
            for acc_label in acc_labels:
                for method in methods:
                    if method in auc_results_tupled[scenario][auc_name][acc_label]:
                        method_values[method].append(auc_results_tupled[scenario][auc_name][acc_label][method]*100)
                    else:
                        method_values[method].append(0)
            
            # Create bar plot
            x = np.arange(len(acc_labels))
            width = 0.15  # Reduced width to fit 5 bars
            
            for j, method in enumerate(methods):
                ax.bar(x + j*width - width*2, method_values[method], width, label=method)
            
            ax.set_xlabel('Accuracy Thresholds')
            ax.set_ylabel('AUC Value')
            ax.set_title(f'{auc_name}')
            ax.set_xticks(x)
            ax.set_xticklabels(acc_labels)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(len(auc_names), 4):
            row, col = i // 2, i % 2
            axes[row, col].set_visible(False)
        
        
        plt.tight_layout()
        os.makedirs('plots6', exist_ok=True)
        plt.savefig(f'plots6/auc_plot_{scenario}.png', dpi=300, bbox_inches='tight')
        #plt.show()

   # Save the results to a JSON file
    with open('auc_results_by_scenario.json', 'w') as f:
        json.dump(auc_results_by_scenario, f, indent=2)


    audio_embed_aucs = json.load(open("angle_range_auc_results_audio_embed_doa.json", "r"))
    gradient_blending_aucs = json.load(open("angle_range_auc_results_gb-va.json", "r"))
    # audio_embed_corrupted_aucs = json.load(open("angle_range_auc_results_audio_embed_corrupted_vision.json", "r"))
    # corrupted_aucs = json.load(open("angle_range_auc_results_corrupted_vision.json", "r"))
    # audio_embed_corrupted_aucs = json.load(open("angle_range_auc_results_corrupted_vision+audio_test_only.json", "r"))
    # corrupted_aucs = json.load(open("angle_range_auc_results_corrupted_vision_test_only.json", "r"))
    video_aucs = json.load(open("angle_range_auc_results.json", "r"))
    audio_only_aucs = json.load(open("angle_range_auc_results_audio_only.json", "r"))
    baseline_mean_aucs = json.load(open("angle_range_auc_results_baseline_mean.json", "r"))
    angle_range_aucs = {
        "audio_embed_doa": audio_embed_aucs,
        "gradient_blending": gradient_blending_aucs,
        # "audio_embed_corrupted": audio_embed_corrupted_aucs,
        # "corrupted": corrupted_aucs,
        "video": video_aucs,
        "audio_only": audio_only_aucs,
        "baseline_mean": baseline_mean_aucs
    }

    pprint(angle_range_aucs)
    import matplotlib.pyplot as plt

    # Create plots for angle range AUCs
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle('Angle Range AUC Results by Method', fontsize=16)

    # Get angle ranges from the first method
    methods = ['audio_embed_doa', 'gradient_blending', 'video', 'audio_only', 'baseline_mean']
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    angle_ranges = ["0-5","5-10", "10-20", "20+"] #list(angle_range_aucs['audio_embed'].keys())

    auc_types = ['rotation_only_aucs', 'total_aucs']

    # Create plots for each AUC type and metric
    plot_configs = [
        ('rotation_only_aucs', 'auc@5'),
        ('rotation_only_aucs', 'auc@10'),
        ('rotation_only_aucs', 'auc@20'),
        ('total_aucs', 'auc@5'),
        ('total_aucs', 'auc@10'),
        ('total_aucs', 'auc@20')
    ]

    for i, (auc_type, auc_metric) in enumerate(plot_configs):
        row, col = i // 3, i % 3
        ax = axes[row, col]
        
        # Prepare data for plotting
        x = np.arange(len(angle_ranges))
        width = 0.15
        
        for j, method in enumerate(methods):
            values = []
            for angle_range in angle_ranges:
                if (angle_range in angle_range_aucs[method] and 
                    auc_type in angle_range_aucs[method][angle_range] and
                    auc_metric in angle_range_aucs[method][angle_range][auc_type]):
                    values.append(angle_range_aucs[method][angle_range][auc_type][auc_metric] * 100)
                else:
                    values.append(0)
            
            ax.bar(x + j*width - width*2, values, width, label=method, color=colors[j], alpha=0.7)
        
        ax.set_xlabel('Angle Range')
        ax.set_ylabel('AUC Value (%)')
        ax.set_title(f'{auc_type} - {auc_metric}')
        ax.set_xticks(x)
        ax.set_xticklabels(angle_ranges)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs('plots6', exist_ok=True)
    plt.savefig('plots6/angle_range_auc_plot.png', dpi=300, bbox_inches='tight')

