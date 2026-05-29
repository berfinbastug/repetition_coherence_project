# Imports and Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
from math import pi
from tapping_analysis_repo import calculate_tap_onset_deviation
import warnings
warnings.filterwarnings('ignore')


# TAPPING
df_tap = pd.read_csv('/Users/bastugb/Desktop/repetition_coherence_project/all_participants_tap_and_actual_onset_values.csv') 
# Extracting values and creating new columns
df_tap['participant_id'] = df_tap['filename'].str.extract(r'pid(\d+)').astype(int)


# TAPPING DATA PRE-PROCESSING
df_tap['block_idx'] = df_tap['filename'].str.extract(r'block_(\d+)').astype(int)
df_tap['trial_idx'] = df_tap['filename'].str.extract(r'itrial_(\d+)').astype(int)
df_tap['actual_onset_values'] = df_tap['actual_onset_values'].apply(lambda x: ast.literal_eval(x))
df_tap['tap_onset_values'] = df_tap['tap_onset_values'].apply(lambda x: ast.literal_eval(x))
df_tap['percentage'] = df_tap['percentage'].apply(lambda x: round(x, 3))
df_tap['n_tap'] = df_tap['tap_onset_values'].apply(len) # Count number of taps

sample_rate = 44100
# Convert tap_onset_values (assumed list/array of sample indices) to seconds
df_tap['tap_onset_s'] = df_tap['tap_onset_values'].apply(lambda taps: np.array(taps) / sample_rate)
df_tap['actual_onset_s'] = df_tap['actual_onset_values'].apply(lambda onsets: np.array(onsets) / sample_rate)

# %%
good_df = df_tap[(df_tap['percentage'] == 1) & (df_tap['unit_dur'] == 0.7) & (df_tap['block_idx'] == 5) & (df_tap['trial_idx'] == 17) & (df_tap['participant_id']== 10)]
bad_df = df_tap[(df_tap['percentage'] == 0.222) & (df_tap['unit_dur'] == 1.0) & (df_tap['block_idx'] == 5) & (df_tap['trial_idx'] == 17) & (df_tap['participant_id']== 1)]


# %%
for i in range(1):
    trial = good_df.iloc[i]
    tap_onset_s = trial['tap_onset_s']
    actual_onset_s = trial['actual_onset_s']

    tap_deviations = calculate_tap_onset_deviation(tap_onset_s, actual_onset_s)
    tapping_phases = (tap_deviations / trial['unit_dur']) * 2 * np.pi

    # Histogram
    num_bins = 16
    counts, bin_edges = np.histogram(tapping_phases, bins=num_bins, range=(0, 2*np.pi))
    angles = bin_edges[:-1]
    width = (2 * np.pi) / num_bins
    max_count = max(counts)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(0.8,0.8))

    ax.bar(angles, counts, width=width, bottom=0.0, align='edge',
           color='#0343DF', edgecolor='#0343DF', alpha=1)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    # Outer ring only
    ax.set_yticks([max_count])
    ax.set_yticklabels([])

    # ---- SAFE π tick labels ----
    # theta_ticks = [0.5*np.pi, np.pi, 1.5*np.pi, 2*np.pi]
    # theta_labels = ["π/2", "π", "3π/2", "2π"]

    # ax.set_xticks(theta_ticks)
    # ax.set_xticklabels(theta_labels, fontsize=10)
    ax.set_xticklabels([])   # removes the π, π/2, 3π/2, 2π labels
    ax.set_yticklabels([])   # removes radial labels (you already have this)

    # # Padding added correctly here
    # ax.tick_params(axis='x', pad=-1)

    fig.savefig('good_trial_circular.svg', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()

# %%
