#=====================
# IMPORT IMPORTANT STUFF
#=====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


#=====================
# FUNCTIONS
#=====================
def weibull_psychometric(x, alpha, beta, gamma, lambd):
    """
    Weibull psychometric function with lapse rate.

    Parameters:
    x : array_like
        Stimulus intensity (can be scalar or array).
    alpha : float
        Threshold (scale parameter).
    beta : float
        Slope (shape parameter).
    gamma : float
        Guess rate (lower asymptote).
    lambd : float
        Lapse rate (1 - upper asymptote).

    Returns:
    y : array_like
        Response probability for each x.
    """
    x = np.asarray(x)
    exponent = -1 * (x / alpha) ** beta
    y = gamma + (1 - gamma - lambd) * (1 - np.exp(exponent))
    return y



# Negative log-likelihood function
def neg_log_likelihood(params, x, n_trials, n_yes):
    alpha, beta, gamma, lambd = params
    p_yes = weibull_psychometric(x, alpha, beta, gamma, lambd)
    # Avoid log(0) by clipping probabilities
    p_yes = np.clip(p_yes, 1e-6, 1 - 1e-6)
    # Binomial log-likelihood
    log_likelihood = n_yes * np.log(p_yes) + (n_trials - n_yes) * np.log(1 - p_yes)
    return -np.sum(log_likelihood)



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



# %%
# all individual's data
# all unit durations are merged
percentage_group_data_pid = df_det_no_outlier.groupby(['percentage', 'participant_id']).agg(
    mean_yes=('actual_response', 'mean'),
    std_yes=('actual_response', 'std'),
    n_yes = ('actual_response', 'sum'),
    n_trial = ('actual_response', 'count'),
).reset_index()



# all conditions are averaged
# all durations are merged
percentage_group_data = df_det_no_outlier.groupby(['percentage']).agg(
    mean_yes=('actual_response', 'mean'),
    std_yes=('actual_response', 'std'),
    n_yes = ('actual_response', 'sum'),
    n_trial = ('actual_response', 'count'),
).reset_index()



# %%
#=====================
# FIT WEIBULL, UNIT DURATIONS MERGED
#=====================
x_fit = np.linspace(0, 1, 500)

x_data_group = percentage_group_data['percentage'].values
y_data_group = percentage_group_data['n_yes'].values
n_trials_group = percentage_group_data['n_trial'].values

# --- Initial parameters and bounds ---
init_params = [0.5, 3.0, 0.2, 0]  # alpha, beta, gamma, lambd
bounds = [(0.01, 1), (0.01, 50), (0, 1), (0, 1)]

res = minimize(neg_log_likelihood, init_params,
    args=(x_data_group, n_trials_group, y_data_group),
    bounds=bounds, method='Powell')

if res.success:
    alpha_fit, beta_fit, gamma_fit, lambd_fit = res.x

    y_fit = weibull_psychometric(x_fit, alpha_fit, beta_fit, gamma_fit, lambd_fit)



# %%
#=====================
# FIT WEIBULL, UNIT DURATIONS SEP
#=====================
# averaged data but this time averaged by unitdur as well
summary_all = df_det_no_outlier.groupby(['unitdur', 'percentage']).agg(
    mean_yes=('actual_response', 'mean'),
    std_yes=('actual_response', 'std'),
    n_yes = ('actual_response', 'sum'),
    n_trial = ('actual_response', 'count'),
).reset_index()


# this is to fit everyone 
summary_participant = df_det_no_outlier.groupby(['unitdur', 'percentage', 'participant_id']).agg(
    mean_yes=('actual_response', 'mean'),
    std_yes=('actual_response', 'std'),
    sum_yes = ('actual_response', 'sum'),
    n_trial = ('actual_response', 'count'),
).reset_index()



unit_durations = [0.4, 0.7, 1.0]
x_fit = np.linspace(0, 1, 500)


all_fit_results = []
for unitdur in unit_durations:

    #=====================
    # FIT WEIBULL, UNIT DURATIONS SEP
    #=====================
    group_data = summary_all[summary_all['unitdur'] == unitdur]
    x_data_group = group_data['percentage'].values
    y_data_group = group_data['n_yes'].values
    n_trials_group = group_data['n_trial'].values

    # --- Initial parameters and bounds ---
    if unitdur == 0.4:
        init_params = [0.5, 3.0, 0.2, 0]  # last param = sigma
        bounds = [(0.01, 1), (0.01, 20), (0, 0.5), (0, 0.5)]
    elif unitdur == 0.7:
        init_params = [0.5, 3.0, 0.2, 0]
        bounds = [(0.01, 1), (0.01, 20), (0, 0.5), (0, 0.5)]
    elif unitdur == 1.0:
        init_params = [0.5, 3.0, 0.2, 0.1]
        bounds = [(0.01, 1), (0.01, 20), (0, 0.5), (0, 0.5)]


    # --- Fit Weibull to individual using MLE ---
    res = minimize(neg_log_likelihood, init_params,
            args=(x_data_group, n_trials_group, y_data_group),
            bounds=bounds, method='Powell')

    if not res.success:
        continue


    alpha_fit, beta_fit, gamma_fit, lambd_fit = res.x
    y_fit = weibull_psychometric(x_fit, alpha_fit, beta_fit, gamma_fit, lambd_fit)


    all_fit_results.append({
        'unit_dur': unitdur,
        'alpha': alpha_fit,
        'beta': beta_fit,
        'gamma': gamma_fit,
        'lambda': lambd_fit
    })



# %%
plt.rcParams['font.family'] = 'Arial'

# Create the 1x2 figure with shared y-axis
fig, (ax_merged, ax_sep) = plt.subplots(
    1, 2,
    figsize=(4.5, 2),                # roughly 2× the width of one panel
    sharey=True
)

x_fit = np.linspace(0, 1, 500)

# =====================================================================
# LEFT PANEL: ALL DURATIONS MERGED
# =====================================================================
x_data_group = percentage_group_data['percentage'].values
y_data_group = percentage_group_data['n_yes'].values
n_trials_group = percentage_group_data['n_trial'].values

init_params = [0.5, 3.0, 0.2, 0]
bounds = [(0.01, 1), (0.01, 50), (0, 1), (0, 1)]

res = minimize(neg_log_likelihood, init_params,
    args=(x_data_group, n_trials_group, y_data_group),
    bounds=bounds, method='Powell')

if res.success:
    alpha_fit, beta_fit, gamma_fit, lambd_fit = res.x
    y_fit = weibull_psychometric(x_fit, alpha_fit, beta_fit, gamma_fit, lambd_fit)

ax_merged.plot(x_fit, y_fit, color='black', linewidth=2, label='All durations')

grouped = (
    percentage_group_data_pid
    .groupby(['percentage'])
    .agg(avg=('mean_yes', 'mean'),
         std=('mean_yes', 'std'),
         n_participants=('mean_yes', 'count'))
    .reset_index()
)
grouped['sem'] = grouped['std'] / np.sqrt(grouped['n_participants'])

ax_merged.plot(grouped['percentage'], grouped['avg'], 'o',
               color='black', markersize=3)
ax_merged.fill_between(grouped['percentage'],
                       grouped['avg'] - grouped['sem'],
                       grouped['avg'] + grouped['sem'],
                       color='black', alpha=0.3)

ax_merged.set_xticks([0, 0.5, 1])
ax_merged.set_yticks([0.2, 0.5, 1])
ax_merged.tick_params(axis='both', labelsize=15)

# =====================================================================
# RIGHT PANEL: BY UNIT DURATION
# =====================================================================
unit_durations = [0.4, 0.7, 1.0]
colors = ['blue', 'green', 'red']

all_fit_results = []
for unitdur, color in zip(unit_durations, colors):
    group_data = summary_all[summary_all['unitdur'] == unitdur]
    x_data_group = group_data['percentage'].values
    y_data_group = group_data['n_yes'].values
    n_trials_group = group_data['n_trial'].values

    if unitdur == 1.0:
        init_params = [0.5, 3.0, 0.2, 0.1]
    else:
        init_params = [0.5, 3.0, 0.2, 0]
    bounds = [(0.01, 1), (0.01, 20), (0, 0.5), (0, 0.5)]

    res = minimize(neg_log_likelihood, init_params,
                   args=(x_data_group, n_trials_group, y_data_group),
                   bounds=bounds, method='Powell')

    if res.success:
        alpha_fit, beta_fit, gamma_fit, lambd_fit = res.x
        all_fit_results.append({
            'unit_dur': unitdur,
            'alpha': alpha_fit, 'beta': beta_fit,
            'gamma': gamma_fit, 'lambda': lambd_fit
        })
        y_fit = weibull_psychometric(x_fit, alpha_fit, beta_fit,
                                      gamma_fit, lambd_fit)

    ax_sep.plot(x_fit, y_fit, color=color, linewidth=2, label=f'{unitdur} s')

    grouped = (
        summary_participant[summary_participant['unitdur'] == unitdur]
        .groupby(['percentage'])
        .agg(avg_yes=('mean_yes', 'mean'),
             std_yes=('mean_yes', 'std'),
             n_participants=('mean_yes', 'count'))
        .reset_index()
    )
    grouped['sem_yes'] = grouped['std_yes'] / np.sqrt(grouped['n_participants'])

    ax_sep.plot(grouped['percentage'], grouped['avg_yes'], 'o',
                color=color, markersize=3)
    ax_sep.fill_between(grouped['percentage'],
                        grouped['avg_yes'] - grouped['sem_yes'],
                        grouped['avg_yes'] + grouped['sem_yes'],
                        color=color, alpha=0.3)

ax_sep.axhline(y=0.5, color='black', linestyle='--', linewidth=1, alpha = 0.7)
ax_sep.legend(title='', fontsize=5)
ax_sep.set_xticks([0, 0.5, 1])
ax_sep.tick_params(axis='both', labelsize=15)

plt.tight_layout()
fig.savefig('weibull_combined.svg', dpi=300,
            transparent=True, bbox_inches='tight', pad_inches=0)
plt.show()




# %%
# here threshold correspond to the value passing the 50 percent yes
thresholds_50 = []  # Store (participant_id, unitdur, threshold_percentage)
unit_durations = [0.4, 0.7, 1.0]
x_fit = np.linspace(0, 1, 200)
all_fit_results = []


# Loop through each subplot axis
for unitdur in unit_durations:
    filtered_df = summary_participant[summary_participant['unitdur'] == unitdur]
    participants = filtered_df['participant_id'].unique()
    fit_results = []

    # Plot individual fits
    for pid in participants:
        pdata = filtered_df[filtered_df['participant_id'] == pid]
        x_data = pdata['percentage'].values
        y_data = pdata['sum_yes'].values
        n_trials = pdata['n_trial'].values

        initial_guess = [0.5, 3.0, 0.0, 0.0]
        bounds = [(0.01, 1.0), (0.1, 10.0), (0.0, 0.5), (0.0, 0.5)]

        res = minimize(neg_log_likelihood, initial_guess, args=(x_data, n_trials, y_data),
                       bounds=bounds, method='powell')

        if res.success:
            alpha, beta, gamma, lambd = res.x
            fit_results.append({
                'participant_id': pid,
                'unitdur': unitdur,
                'alpha': alpha,
                'beta': beta,
                'gamma': gamma,
                'lambd': lambd
            })

            y_fit = weibull_psychometric(x_fit, alpha, beta, gamma, lambd)

            # --- Find threshold where y_fit crosses 0.5 ---
            threshold_index = np.argmax(y_fit > 0.5)
            threshold_percentage = x_fit[threshold_index]

            thresholds_50.append({
            'participant_id': pid,
            'unitdur': unitdur,
            'threshold_percentage': threshold_percentage
            })

    # Store fits
    fit_df = pd.DataFrame(fit_results)
    all_fit_results.append(fit_df)

    # Group-level fit using summary
    group_data = summary_all[summary_all['unitdur'] == unitdur]
    x_data_group = group_data['percentage'].values
    y_data_group = group_data['n_yes'].values
    n_trials_group = group_data['n_trial'].values



    res_group = minimize(
        neg_log_likelihood,
        x0=[0.5, 3.0, 0.0, 0.0],
        args=(x_data_group, n_trials_group, y_data_group),
        bounds=[(0.01, 1.0), (0.1, 10.0), (0.0, 1), (0.0, 1)],
        method='powell'
    )



combined_fit_df = pd.concat(all_fit_results, ignore_index=True)
threshold_df = pd.DataFrame(thresholds_50)


threshold_df_det = threshold_df.dropna(subset = ['threshold_percentage'])
# Convert to string for categorical plotting
threshold_df_det['unitdur'] = threshold_df_det['unitdur'].astype(str)



# %%
#=====================
# VIOLIN PLOT OPTION
#=====================
colors = {0.4: "blue", 0.7: "green", 1.0: "red"}
colors_str = {str(k): v for k, v in colors.items()}

# Turn off Seaborn grid styling
sns.set_style("white")
plt.rcParams['font.family'] = 'Arial'
fig, ax = plt.subplots(figsize=(2.5, 2.1))

# Violin plot — distribution shape
sns.violinplot(
    data=threshold_df_det,
    x='unitdur',
    y='threshold_percentage',
    palette=colors_str,
    width=0.7,
    ax=ax,
    inner=None,                  # remove default inner sticks/box; we'll add dots separately
    linewidth=1,                 # outline thickness of the violin
    cut=0,                       # don't extend the KDE past the data range
    saturation=0.8,              # slight desaturation for softer fill
)


# Set violin fill alpha
for violin in ax.collections:
    violin.set_alpha(0.8)        # 0 = transparent, 1 = opaque

# Overlay individual data points
sns.stripplot(
    data=threshold_df_det,
    x='unitdur',
    y='threshold_percentage',
    color='black',
    alpha=0.4,
    size=3,
    jitter=0.12,                 # horizontal jitter (0 = stacked, ~0.2 = wide spread)
    ax=ax,
    edgecolor='white',
    linewidth=0.8,
)

# Optional: add a median line across each violin
medians = threshold_df_det.groupby('unitdur')['threshold_percentage'].median()
for i, (group_name, median_val) in enumerate(medians.items()):
    ax.hlines(median_val, i - 0.25, i + 0.25, color='black', linewidth=2, zorder=10, alpha = 0.6)

# Labels & style
ax.set_xlabel('')
ax.set_ylabel('')
ax.grid(False)
plt.xticks(fontsize=15)
plt.yticks([0, 0.5, 1.0], fontsize=15)
plt.ylim([0, 1])
plt.tight_layout()
fig.savefig('threshold_violin.svg', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
plt.show()
# %%
