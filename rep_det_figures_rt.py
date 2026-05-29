#=====================
# IMPORT IMPORTANT STUFF
#=====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem
import matplotlib.cm as cm
from scipy.stats import zscore
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


#=====================
# READ FILES
#=====================
# REPETITION DETECTION
df_det_common = pd.read_csv('/Users/bastugb/Desktop/repetition_coherence_project/df_det_common.csv')
len(df_det_common['participant_id'].unique())
df_det_common['participant_id'].unique()


# Add cycle info
df_det_common['n_cycle'] = df_det_common['rt'] / df_det_common['unitdur']
len(df_det_common)


#=====================
# OUTLIERS BY RT
#=====================
anticipatory_mask = df_det_common['n_cycle'] >= 1  # this is anticipatory
df_det_common = df_det_common[anticipatory_mask]
len(df_det_common)

df_det_common['is_outlier'] = False

# Group by percentage and unitdur
for (perc, dur), group in df_det_common.groupby(['percentage', 'unitdur']):
    Q1 = group['rt'].quantile(0.25)
    Q3 = group['rt'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    # lower = group['n_tap_filtered'].quantile(0.05)
    # upper = group['n_tap_filtered'].quantile(0.95)
    # Mask for outliers
    outlier_mask = (group['rt'] < lower) | (group['rt'] > upper)
    # outlier_mask = (group['n_tap'] > upper)
    # Mark them in the original DataFrame
    df_det_common.loc[group.index, 'is_outlier'] = outlier_mask

# Filter out the outliers
df_det_clean = df_det_common[~df_det_common['is_outlier']].reset_index(drop=True)

num_outliers = len(df_det_common) - len(df_det_clean)
percentage_outliers = 100 * num_outliers / len(df_det_common)
print(f"{num_outliers} outliers removed ({percentage_outliers:.2f}% of total)")



#=====================
# PLOT OUTLIER ANALYSIS
#=====================
# Define custom color palette
custom_palette = {0.4: "#15619cf6", 0.7: "#3bbc3bf8", 1.0: "#e22f2ff8"}

plt.figure(figsize=(12, 6))

# Create a grouped boxplot using the custom palette
sns.boxplot(data=df_det_common, x='percentage', y='rt', hue='unitdur',
            palette=custom_palette, showfliers=False)

# Overlay outlier RTs using the same palette
outliers = df_det_common[df_det_common['is_outlier']]
sns.stripplot(data=outliers, x='percentage', y='rt', hue='unitdur',
              marker='o', dodge=True, palette=custom_palette,
              edgecolor='black', linewidth=0.5, size=5)

# Clean up the legend (to remove duplication from stripplot)
handles, labels = plt.gca().get_legend_handles_labels()
n_unique = df_det_common['unitdur'].nunique()
plt.legend(handles[:n_unique], labels[:n_unique], title='Unit Duration')

# Plot formatting
plt.title('RTs by Percentage and Unit Duration (Dots are outliers)')
plt.ylabel('Reaction Time (s)')
plt.xlabel('Repetition Coherence')
plt.grid(True, axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()



#=========================
# OUTLIERS BY PERFORMANCE
#=========================
# Calculate mean accuracy per participant-block
perf = df_det_clean.groupby(['participant_id', 'block_idx'])['correct'].mean().reset_index()
perf.rename(columns={'correct': 'block_accuracy'}, inplace=True)

# Compute z-scores for block accuracy
perf['z_score'] = zscore(perf['block_accuracy'])

# Mark bad performance blocks
threshold = 3
perf['bad_performance'] = perf['z_score'].abs() > threshold

# Identify bad blocks
bad_blocks = perf[perf['bad_performance']][['participant_id', 'block_idx']]
print(bad_blocks)



#=====================
# FINALIZE OUTLIER ANALYSIS
#=====================
# Merge and filter out bad blocks from the main data
df_det_no_outlier = df_det_clean.merge(bad_blocks, on=['participant_id', 'block_idx'], how='left', indicator=True)
df_det_no_outlier = df_det_no_outlier[df_det_no_outlier['_merge'] == 'left_only'].drop(columns=['_merge'])

# check how much data remains
print(f"Original trials: {len(df_det_clean)}")
print(f"Cleaned trials: {len(df_det_no_outlier)}")
print(f"difference: {len(df_det_clean) - len(df_det_no_outlier)}")



# Group and compute required metrics including MADs
participant_summary_ncycle = (
    df_det_no_outlier
    .groupby(['participant_id', 'percentage', 'unitdur', 'correct'])
    .agg(
        sum_correct=('correct', 'sum'),
        median_ncycle=('n_cycle', 'median'),
        median_rt=('rt', 'median'),
    )
    .reset_index()
)


correct_trials = df_det_no_outlier[df_det_no_outlier['correct'] == 1]


# Define a custom MAD function
def mad(series):
    return np.median(np.abs(series - np.median(series)))

# Step 2: Group and compute required metrics including MADs
participant_summary_correct_ncycle = (
    correct_trials
    .groupby(['participant_id', 'percentage', 'unitdur'])
    .agg(
        sum_correct=('correct', 'sum'),
        median_ncycle=('n_cycle', 'median'),
        mad_ncycle=('n_cycle', mad),
        median_rt=('rt', 'median'),
        mad_rt=('rt', mad)
    )
    .reset_index()
)


# %%
# =====================================================================
# Prepare data (done once)
# =====================================================================
plot_data = participant_summary_correct_ncycle[
    participant_summary_correct_ncycle['percentage'] >= 0.111].copy()
plot_data['percentage'] = plot_data['percentage'].round(3)
plot_data['unitdur'] = plot_data['unitdur'].astype(str)

base_colors = {'0.4': (0, 0, 1), '0.7': (0, 0.6, 0), '1.0': (1, 0, 0)}

plt.rcParams['font.family'] = 'Arial'

# %%
# =====================================================================
# Helper to plot one panel (avoids duplicating the loop)
# =====================================================================
def plot_panel(ax, grouped, y_col, sem_col, ylabel, yticks, xticks):
    # normalize correct for line transparency
    min_c = grouped['mean_correct'].min()
    max_c = grouped['mean_correct'].max()
    grouped = grouped.copy()
    grouped['correct_norm'] = (grouped['mean_correct'] - min_c) / (max_c - min_c)

    for dur in ['0.4', '0.7', '1.0']:
        subset = grouped[grouped['unitdur'] == dur].sort_values('percentage')
        x_vals = subset['percentage']
        y_vals = subset[y_col]
        y_err = subset[sem_col]

        # SEM band
        ax.fill_between(x_vals, y_vals - y_err, y_vals + y_err,
                        color=base_colors[dur], alpha=0.2)

        # line segments shaded by accuracy
        for i in range(len(subset) - 1):
            x_pair = x_vals.iloc[i:i+2]
            y_pair = y_vals.iloc[i:i+2]
            alpha = subset['correct_norm'].iloc[i]
            ax.plot(x_pair, y_pair, color=base_colors[dur],
                    alpha=alpha, linewidth=2)

    # ax.set_xlabel('Coherence', fontsize=70, labelpad=10)
    # ax.set_ylabel(ylabel, fontsize=70, labelpad=8)
    for spine in ['left', 'bottom', 'top', 'right']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color('black')
        ax.spines[spine].set_linewidth(1)
    ax.set_yticks(yticks)
    ax.set_xticks(xticks)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=15)
    ax.grid(False)


# %%
# =====================================================================
# Build the 1x2 figure
# =====================================================================
fig, (ax_rt, ax_ncycle) = plt.subplots(
    1, 2, figsize=(7, 1.8), facecolor='white',
    gridspec_kw={'wspace': 0.6}        # bigger = more gap between panels
)

# --- Left panel: RT ---
grouped_rt = plot_data.groupby(['percentage', 'unitdur']).agg(
    mean_rt=('median_rt', 'mean'),
    sem_rt=('median_rt', sem),
    mean_correct=('sum_correct', 'mean')
).reset_index().rename(columns={'mean_rt': 'y', 'sem_rt': 'sem'})

plot_panel(ax_rt, grouped_rt, 'y', 'sem', 'RT', [2, 4, 6], [0.25, 0.5, 0.75, 1])

# --- Right panel: N Cycles ---
grouped_ncycle = plot_data.groupby(['percentage', 'unitdur']).agg(
    mean_ncycle=('median_ncycle', 'mean'),
    sem_ncycle=('median_ncycle', sem),
    mean_correct=('sum_correct', 'mean')
).reset_index().rename(columns={'mean_ncycle': 'y', 'sem_ncycle': 'sem'})

plot_panel(ax_ncycle, grouped_ncycle, 'y', 'sem', 'N Cycles', [4, 8, 12], [0.25, 0.5, 0.75, 1])

fig.savefig('ncycle_rt_combined.svg', dpi=300, transparent=True,
            bbox_inches='tight', pad_inches=0)
plt.show()
# %%
