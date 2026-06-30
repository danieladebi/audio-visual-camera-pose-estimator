import json
import numpy as np
import os
import matplotlib.pyplot as plt
from pprint import pprint


if __name__ == "__main__":
    vision_only_angle_range_aucs = json.load(open("angle_range_auc_results_vision_only_NEW.json", "r"))
    ours_angle_range_aucs = json.load(open("angle_range_auc_results_ours_NEW.json", "r"))

    # Collect AUC@20, AUC@10, and AUC@5
    vision_only_auc_20s = {"0-5": [], "5-10": [], "10-20": [], "20+": []}
    ours_auc_20s = {"0-5": [], "5-10": [], "10-20": [], "20+": []}
    vision_only_auc_10s = {"0-5": [], "5-10": [], "10-20": [], "20+": []}
    ours_auc_10s = {"0-5": [], "5-10": [], "10-20": [], "20+": []}
    vision_only_auc_5s = {"0-5": [], "5-10": [], "10-20": [], "20+": []}
    ours_auc_5s = {"0-5": [], "5-10": [], "10-20": [], "20+": []}

    for angle_range in vision_only_angle_range_aucs.keys():
        vision_only_aucs = vision_only_angle_range_aucs[angle_range]
        ours_aucs = ours_angle_range_aucs[angle_range]

        for auc_category in vision_only_aucs.keys():
            if "total" in auc_category:
                vision_only_auc = vision_only_aucs[auc_category]
                ours_auc = ours_aucs[auc_category]

                # AUC@20
                if "auc@20" in vision_only_auc and "auc@20" in ours_auc:
                    vision_only_auc_20s[angle_range] = vision_only_auc["auc@20"]
                    ours_auc_20s[angle_range] = ours_auc["auc@20"]

                # AUC@10
                if "auc@10" in vision_only_auc and "auc@10" in ours_auc:
                    vision_only_auc_10s[angle_range] = vision_only_auc["auc@10"]
                    ours_auc_10s[angle_range] = ours_auc["auc@10"]

                # AUC@5
                if "auc@5" in vision_only_auc and "auc@5" in ours_auc:
                    vision_only_auc_5s[angle_range] = vision_only_auc["auc@5"]
                    ours_auc_5s[angle_range] = ours_auc["auc@5"]

    # Plot AUC@10
    labels10 = list(vision_only_auc_10s.keys())
    vision_vals_10 = [v if isinstance(v, (int, float)) else np.nan for v in vision_only_auc_10s.values()]
    ours_vals_10 = [v if isinstance(v, (int, float)) else np.nan for v in ours_auc_10s.values()]

    x10 = np.arange(len(labels10))
    width = 0.25  # narrower bars
    width10 = width

    plt.figure(figsize=(8, 4))
    plt.bar(x10 - width10/2, vision_vals_10, width10, label="Vision Only", alpha=0.7)
    plt.bar(x10 + width10/2, ours_vals_10, width10, label="Ours", alpha=0.7)

    plt.xticks(x10, labels10)
    plt.ylabel("AUC@10")
    plt.title("AUC@10 per Angle Range in Degrees", fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig("visualize_auc10_per_angle_range_histogram_NEW_total.png")

    # Plot AUC@5
    labels5 = list(vision_only_auc_5s.keys())
    vision_vals_5 = [v if isinstance(v, (int, float)) else np.nan for v in vision_only_auc_5s.values()]
    ours_vals_5 = [v if isinstance(v, (int, float)) else np.nan for v in ours_auc_5s.values()]

    x5 = np.arange(len(labels5))
    width5 = width  # same narrower width

    plt.figure(figsize=(8, 4))
    plt.bar(x5 - width5/2, vision_vals_5, width5, label="Vision Only", alpha=0.7)
    plt.bar(x5 + width5/2, ours_vals_5, width5, label="Ours", alpha=0.7)

    plt.xticks(x5, labels5)
    plt.ylabel("AUC@5")
    plt.title("AUC@5 per Angle Range in Degrees", fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig("visualize_auc5_per_angle_range_histogram_NEW_total.png")

    # Prepare data
    labels = list(vision_only_auc_20s.keys())
    vision_vals = [v if isinstance(v, (int, float)) else np.nan for v in vision_only_auc_20s.values()]
    ours_vals = [v if isinstance(v, (int, float)) else np.nan for v in ours_auc_20s.values()]

    # Bar-style histogram
    x = np.arange(len(labels))

    plt.figure(figsize=(8, 4))
    plt.bar(x - width/2, vision_vals, width, label="Vision Only", alpha=0.7)
    plt.bar(x + width/2, ours_vals, width, label="Ours", alpha=0.7)

    plt.xticks(x, labels)
    plt.ylabel("AUC@20")
    plt.title("AUC@20 per Angle Range in Degrees", fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig("visualize_auc20_per_angle_range_histogram_NEW_total.png")