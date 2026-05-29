# Imports and Setup
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.stats import sem


import warnings
warnings.filterwarnings('ignore')



# TAPPING
df_tap_common = pd.read_csv('/Users/bastugb/Desktop/repetition_coherence_project/df_tap_common.csv') 

len(df_tap_common['participant_id'].unique())

# # TAPPING
# df_tap = pd.read_csv('/Users/bastugb/Desktop/repetition_coherence_project/df_tap_common.csv') 
# # Extracting values and creating new columns
# df_tap['participant_id'] = df_tap['filename'].str.extract(r'pid(\d+)').astype(int)
len(df_tap_common['participant_id'].unique())


df_tap_common['block_idx'] = df_tap_common['filename'].str.extract(r'block_(\d+)').astype(int)
df_tap_common['trial_idx'] = df_tap_common['filename'].str.extract(r'itrial_(\d+)').astype(int)
df_tap_common['actual_onset_values'] = df_tap_common['actual_onset_values'].apply(lambda x: ast.literal_eval(x))
df_tap_common['tap_onset_values'] = df_tap_common['tap_onset_values'].apply(lambda x: ast.literal_eval(x))
df_tap_common['percentage'] = df_tap_common['percentage'].apply(lambda x: round(x, 3))
df_tap_common['n_tap'] = df_tap_common['tap_onset_values'].apply(len) # Count number of taps


len(df_tap_common)


# Convert tap_onset_values (assumed list/array of sample indices) to seconds
sample_rate = 44100
df_tap_common['tap_onset_s'] = df_tap_common['tap_onset_values'].apply(lambda taps: np.array(taps) / sample_rate)
df_tap_common['actual_onset_s'] = df_tap_common['actual_onset_values'].apply(lambda onsets: np.array(onsets) / sample_rate)


original_len = len(df_tap_common)

# Filter out rows where 'tap_onset_s' has 1 or fewer values
df_tap_common = df_tap_common[
    df_tap_common['tap_onset_s'].apply(lambda x: isinstance(x, (list, np.ndarray)) and len(x) > 2)
]

removed = original_len - len(df_tap_common)
print(f"Removed {removed} rows with 1 or fewer tap_onset_s values.")



df_tap_common['is_outlier'] = False

# Group by percentage and unitdur
for (perc, dur), group in df_tap_common.groupby(['percentage', 'unit_dur']):
    Q1 = group['n_tap'].quantile(0.25)
    Q3 = group['n_tap'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    # lower = group['n_tap_filtered'].quantile(0.05)
    # upper = group['n_tap_filtered'].quantile(0.95)
    # Mask for outliers
    outlier_mask = (group['n_tap'] < lower) | (group['n_tap'] > upper)
    # outlier_mask = (group['n_tap'] > upper)
    # Mark them in the original DataFrame
    df_tap_common.loc[group.index, 'is_outlier'] = outlier_mask

# Filter out the outliers
df_tap_clean = df_tap_common[~df_tap_common['is_outlier']].reset_index(drop=True)

num_outliers = len(df_tap_common) - len(df_tap_clean)
percentage_outliers = 100 * num_outliers / len(df_tap_common)
print(f"{num_outliers} outliers removed ({percentage_outliers:.2f}% of total)")



from tapping_analysis_repo import compute_sliding_window_values
df_tap_clean[['slide_resultant_length', 'slide_rayleigh_z', 'z_scored_r']] = df_tap_clean.apply(
    lambda row: pd.Series(compute_sliding_window_values(row['actual_onset_s'], row['tap_onset_s'], row['unit_dur'])),
    axis=1
)


# Organize .... lists by (participant, unitdur, percentage)
aligned_data = {}  # Dictionary to store progressions

for (participant_id, unitdur, percentage), group in df_tap_clean.groupby(['participant_id', 'unit_dur', 'percentage']):
    zvals = []
    
    for z_list in group['z_scored_r']:
        zvals.append(z_list)  # Store lists

    aligned_data[(participant_id, unitdur, percentage)] = zvals  # Store per participant

# Find the max sequence length (to pad shorter ones)
max_length = max(max(len(trial) for trial in trials) for trials in aligned_data.values())

# Convert to DataFrame, padding shorter trials with NaN
df_avg_zvals = {key: np.nanmean([np.pad(trial, (0, max_length - len(trial)), constant_values=np.nan) 
                               for trial in trials], axis=0) 
              for key, trials in aligned_data.items()}

# that's grouped by participant id, percentage, and unit dur.
df_avg_zvals = pd.DataFrame(df_avg_zvals)  # Convert to DataFrame


# this is grouped by unit dur and percentage only. 
# averages of averages
df_avg_zvals.columns = pd.MultiIndex.from_tuples(df_avg_zvals.columns, names=['participant', 'unit_dur', 'percentage'])
grouped = df_avg_zvals.groupby(level=['unit_dur', 'percentage'], axis=1)
df_mean_zvals = grouped.aggregate(np.nanmean)



# Define the exponential saturation model
def exp_saturation(x, a, b, c):
    return a * (1 - np.exp(-b * x)) + c



# Store all fits per participant for later averaging
all_fits = []

x_data = np.arange(max_length)

for key, trials in aligned_data.items():
    participant_id, unitdur, percentage = key

    # Average across trials for this participant
    avg_z = np.nanmean([np.pad(trial, (0, max_length - len(trial)), constant_values=np.nan)
                        for trial in trials], axis=0)

    mask = ~np.isnan(avg_z)
    if np.sum(mask) > 5:  # Require some data points
        try:
            popt, _ = curve_fit(exp_saturation, x_data[mask], avg_z[mask], p0=[1, 0.1, 0], maxfev=10000)
            all_fits.append({
                'participant_id': participant_id,
                'unit_dur': unitdur,
                'percentage': percentage,
                'a': popt[0],
                'b': popt[1],
                'c': popt[2]
            })
        except RuntimeError:
            pass


df_fits = pd.DataFrame(all_fits)

summary_df_fits = df_fits.groupby(['unit_dur', 'percentage']).agg(
    mean_b=('b', 'mean'),
    std_b=('b', 'std')
).reset_index()


p_threshold = 0.05
z_threshold = norm.ppf(1 - p_threshold)  # One-sided test



# %%
sns.set(style="white")
# 20 sliding windows in total, for all conditions
sliding_window_indices = np.arange(21)  
# Define x values for smooth curves
x_fit = np.linspace(0, max_length - 1, 500)

# Plot average fit for each condition
fig, axs = plt.subplots(1, 3, figsize=(8.2, 2.5), sharey=True)
durations = sorted(df_fits['unit_dur'].unique())
colormaps = {0.4: cm.Blues, 0.7: cm.Greens, 1.0: cm.Reds}


for i, dur in enumerate(durations):
    
    ax = axs[i]
    cmap = colormaps[dur]

    # Normalize percentage values to [0, 1] for gradient mapping
    percentages = sorted(df_fits['percentage'].unique())
    normalizee = Normalize(vmin=min(percentages), vmax=max(percentages))

    for perc in percentages:

        # Access data from multi-index DataFrame
        actual_series = df_mean_zvals[(dur, perc)].values  # shape = (21,)
        subset = df_fits[(df_fits['unit_dur'] == dur) & (df_fits['percentage'] == perc)]

        
        if not subset.empty:
            y_fits = []
            for _, row in subset.iterrows():
                y_fits.append(exp_saturation(x_fit, row['a'], row['b'], row['c']))
            
            y_fits = np.array(y_fits)
            mean_fit = np.nanmean(y_fits, axis=0)
            sem_fit = sem(y_fits, axis=0)

            # Assign color based on percentage gradient
            color = cmap(normalizee(perc))

            ax.plot(x_fit, mean_fit, color=color, label=f'{perc:.2f}', lw=2)
            ax.fill_between(x_fit, mean_fit - sem_fit, mean_fit + sem_fit, color=color, alpha=0.2)

            ax.plot(sliding_window_indices, actual_series, 'o', color=color, markersize=2, alpha=0.8)

    # ax.set_title(f'{dur} s', fontsize=70, pad = 25)
    # ax.set_xlabel('Sliding Window Index', fontsize=18)
    # if i == 0:
    #     ax.set_ylabel('Rayleigh-z', fontsize=18)

    ax.tick_params(labelsize=15)
    ax.set_ylim(bottom=-0.5)
    ax.axhline(z_threshold, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    # ax.text(x_fit[-1], z_threshold + 0.1, f'p < {p_threshold}', color='gray', fontsize=12, ha='right')


    # LEGEND AESTHETICS
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    # cb = ColorbarBase(cax, cmap=cmap, norm=normalizee)
    cb = ColorbarBase(cax, cmap=cmap, norm=normalizee, ticks=[0, 0.5, 1])

    
    # Show label only for the rightmost subplot
    # if i == len(durations) - 1:
    #     cb.set_label('Coherence', fontsize=40)
    cb.ax.tick_params(labelsize=12)



# --- Shared Axis Labels ---
# fig.text(0.5, 0.05, 'Sliding Window Index', ha='center', fontsize=70)
# fig.text(0.08, 0.53, 'Consistency (z)', va='center', rotation='vertical', fontsize=70)

         
plt.tight_layout(rect=[0.1, 0.15, 1, 1])  # Adjust layout to leave space for labels
plt.subplots_adjust(wspace=0.35)  # or try 0.5, 0.6 for more space
fig.savefig('exp_saturation.svg', dpi=300, transparent=True,
            bbox_inches='tight', pad_inches=0)
plt.show()

# %%
