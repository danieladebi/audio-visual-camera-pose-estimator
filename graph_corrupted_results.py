import seaborn as sns 
import pandas as pd
import seaborn as sns
import atexit


vision_only_auc_20s = {100: 3.66, 90: 5.486, 80:6.6786, 70: 7.0089, 60: 7.17698, 50:7.257, 0:7.33}
slfm_vision_auc_20s = {100: 4.42, 90: 6.018, 80: 7.7759, 70: 8.1478, 60: 8.319, 50: 8.37, 0: 8.41}
doa_360_vision_auc_20s = {100: 4.48, 90: 5.62296, 80: 7.5126, 70: 7.993, 60: 8.2148, 50: 8.314, 0: 8.34}
doa_360_slfm_vision_auc_20s = {100: 4.80, 90: 6.52812, 80: 8.1926855, 70: 8.434, 60: 8.63, 50: 8.73, 0: 8.82}
audio_embed_vision_auc_20s = {100: 4.36, 90: 5.658, 80: 7.265, 70: 7.709, 60: 7.898, 50: 7.998, 0: 8.07}

vision_only_auc_10s = {100: 1.00, 90: 1.60, 80: 2.04, 70: 2.17856, 60: 2.249, 50: 2.29, 0: 2.33}
# slfm_vision_auc_10s = {100: ..., 90: ..., 80:..., 70: ..., 60: ..., 50: ..., 0: ...}
# doa_360_vision_auc_10s = {100: ..., 90: ..., 80:..., 70: ..., 60: ..., 50: ..., 0: ...}
doa_360_slfm_vision_auc_10s = {100: 1.44, 90: 1.798, 80: 2.59, 70: 2.817, 60: 2.8955, 50: 2.94, 0: 2.99}
# audio_embed_vision_auc_10s = {100: ..., 90: ..., 80:..., 70: ..., 60: ..., 50: ..., 0: ...}

# Plot all *_auc_20s dicts on one line chart
def plot_all_auc_20s():
    sns.set_theme(style="whitegrid", rc={"lines.markersize": 10})

    name_map = {
        'vision_only_auc_20s': 'Reloc3r',
        "audio_embed_vision_auc_20s": "Reloc3r w/ Naive Audio",
        'slfm_vision_auc_20s': 'Ours w/o DOA',
        "doa_360_vision_auc_20s": 'Ours w/o SLfM',
        'doa_360_slfm_vision_auc_20s': "Ours",
    }

    marker_styles = {
        "Reloc3r": "o",
        "Reloc3r w/ Naive Audio": "s",
        "Ours w/o DOA": "^",
        "Ours w/o SLfM": "v",
        "Ours": "D"
    }

    datasets = []
    for var_name, var_val in globals().items():
        if var_name.endswith('_auc_20s') and isinstance(var_val, dict):
            label = name_map.get(var_name, var_name.replace('_auc_20s', '').replace('_', ' ').title())
            datasets.append((label, var_val))

    rows = []
    for label, d in datasets:
        for cl, auc in d.items():
            rows.append({'corruption_level': int(cl), 'AUC': float(auc), 'Model': label})

    if not rows:
        raise ValueError("No *_auc_20s datasets found to plot.")

    df = pd.DataFrame(rows).sort_values(['Model', 'corruption_level'])

    ax = sns.lineplot(
        data=df,
        x='corruption_level',
        y='AUC',
        hue='Model',
        style='Model',
        markers=marker_styles,
        dashes=False,
    )

    ax.set_title('AUC@20 vs Corruption %')
    ax.set_xlabel('Corruption %')
    ax.set_ylabel('AUC@20')
    ax.set_xlim(0, 100)
    ax.legend(title='Model')

    # Ensure all markers enlarged (in case backend ignores rc)
    for line in ax.lines:
        line.set_markersize(8)

    return ax

def plot_all_auc_10s():

    name_map = {
        'vision_only_auc_10s': 'Reloc3r',
        'doa_360_slfm_vision_auc_10s': 'Ours',
    }
    marker_styles = {
        "Reloc3r": "o",
        "Ours": "D",
    }

    rows = []
    for var_name, var_val in globals().items():
        if var_name.endswith('_auc_10s') and isinstance(var_val, dict):
            label = name_map.get(var_name, var_name.replace('_auc_10s', '').replace('_', ' ').title())
            for cl, auc in var_val.items():
                rows.append({'corruption_level': int(cl), 'AUC': float(auc), 'Model': label})

    if not rows:
        raise ValueError("No *_auc_10s datasets found to plot.")

    df = pd.DataFrame(rows).sort_values(['Model', 'corruption_level'])
    sns.set_theme(style="whitegrid", rc={"lines.markersize": 10})
    ax = sns.lineplot(
        data=df,
        x='corruption_level',
        y='AUC',
        hue='Model',
        style='Model',
        markers=marker_styles,
        dashes=False,
    )
    ax.set_title('AUC@10 vs Corruption %')
    ax.set_xlabel('Corruption %')
    ax.set_ylabel('AUC@10')
    ax.set_xlim(0, 100)
    ax.legend(title='Model')
    for line in ax.lines:
        line.set_markersize(8)
    return ax

def _save_auc10s():
    ax10 = plot_all_auc_10s()
    fig10 = ax10.get_figure()
    fig10.tight_layout()
    fig10.savefig("corruption_results_auc10s.png", dpi=200)
    atexit.register(_save_auc10s)

if __name__ == "__main__":
    # ax = plot_all_auc_20s()
    # fig = ax.get_figure()
    # fig.tight_layout()
    # fig.savefig("corruption_results.png", dpi=200)

    _save_auc10s()
