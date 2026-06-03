# Imports and Setup
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.optimize import minimize


import warnings
warnings.filterwarnings('ignore')



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
def neg_log_likelihood(params, x, n_trials, n_significant):
    alpha, beta, gamma, lambd = params
    p_sig = weibull_psychometric(x, alpha, beta, gamma, lambd)
    # Avoid log(0) by clipping probabilities
    p_sig = np.clip(p_sig, 1e-6, 1 - 1e-6)
    # Binomial log-likelihood
    log_likelihood = n_significant * np.log(p_sig) + (n_trials - n_significant) * np.log(1 - p_sig)
    return -np.sum(log_likelihood)


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



# Prepare the result list for storing delays
tap_onset_delays_list = []
itis_list = []
# Iterate over each row in the DataFrame
for index, row in df_tap_clean.iterrows():
    # Select the last 20 presentation cycles (actual_onset_s)
    actual_window = row['actual_onset_s'][-20:]
    
    # Select tap onsets (tap_onset_s) that fall within the range of the last 20 presentation cycles
    # Define the range as the first and last actual onset in the window
    start_range = actual_window[0]
    # till the end of the trial which includes: time point of presentation, the duration of the presentation, zero pad
    zero_pad = 0.2
    # end_range = actual_window[-1] + row['unit_dur'] + zero_pad
    end_range = actual_window[-1]
    
    # Select tap onsets that are within this range
    tap_window = np.array(row['tap_onset_s'])
    valid_tap_onsets = tap_window[(tap_window >= start_range) & (tap_window <= end_range)]
    iti = np.diff(np.array(valid_tap_onsets))
    itis_list.append(iti)

    # Find the closest indices in actual_window that are less than or equal to the tap onsets using np.searchsorted
    closest_indices = np.searchsorted(actual_window, valid_tap_onsets, side='right') - 1
    
    # Ensure indices are valid (i.e., actual onset is before tap onset)
    valid_indices = (closest_indices >= 0)
    closest_actuals = np.array(actual_window)[closest_indices[valid_indices]]
    valid_tap_onsets = valid_tap_onsets[valid_indices]
    
    # Compute tap onset delays (difference between tap and actual)
    tap_onset_delays = valid_tap_onsets - closest_actuals
    
    # Add the computed delays for this row to the list
    tap_onset_delays_list.append(tap_onset_delays)

# After processing all rows, you can store the delays in the DataFrame or another structure
df_tap_clean['tap_onset_delays_stabil'] = tap_onset_delays_list
df_tap_clean['itis_stabil'] = itis_list



from tapping_analysis_repo import get_tapping_vectors
df_tap_clean[['tapping_phases_stabil', 'tapping_vectors_stabil']] = df_tap_clean.apply(
    lambda row: pd.Series(get_tapping_vectors(row['tap_onset_delays_stabil'], row['unit_dur'])), 
    axis=1
)


from pycircstat2.descriptive import circ_r,circ_dispersion
from pycircstat2.hypothesis import rayleigh_test
# DESCRIPTIVES (LAST 20 CYCLES)
df_tap_clean['resultant_length'] = df_tap_clean['tapping_phases_stabil'].apply(lambda x: circ_r(np.array(x)))


from tapping_analysis_repo import compute_z_scored_r

p_threshold = 0.05
z_threshold = norm.ppf(1 - p_threshold)  # One-sided test

df_tap_clean['z_scored_r_all'] = df_tap_clean.apply(lambda row: compute_z_scored_r(row['resultant_length'], row['n_tap']), axis=1)

df_tap_clean['significant'] = df_tap_clean['z_scored_r_all'].apply(
    lambda z: 1 if pd.notna(z) and z > z_threshold else 0
)


summary_participant = df_tap_clean.groupby(['unit_dur', 'percentage', 'participant_id']).agg(
    mean_sig=('significant', 'mean'),
    std_sig=('significant', 'std'),
    n_sig = ('significant', 'sum'),
    n_trial = ('significant', 'count'),
).reset_index()


summary_all = df_tap_clean.groupby(['unit_dur', 'percentage']).agg(
    avg_sig=('significant', 'mean'),
    std=('significant', 'std'),
    n_sig=('significant', 'sum'),
    n_trial=('significant', 'count'),
).reset_index()


# here threshold correspond to the value passing the 50 percent yes
thresholds_50 = []  # Store (participant_id, unitdur, threshold_percentage)

unit_durations = [0.4, 0.7, 1.0]

x_fit = np.linspace(0, 1, 200)
all_fit_results = []

# Loop through each subplot axis
for unitdur in unit_durations:
    filtered_df = summary_participant[summary_participant['unit_dur'] == unitdur]
    participants = filtered_df['participant_id'].unique()
    fit_results = []

    # Plot individual fits
    for pid in participants:
        pdata = filtered_df[filtered_df['participant_id'] == pid]
        x_data = pdata['percentage'].values
        y_data = pdata['n_sig'].values
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
    group_data = summary_all[summary_all['unit_dur'] == unitdur]
    x_data_group = group_data['percentage'].values
    y_data_group = group_data['n_sig'].values
    n_trials_group = group_data['n_trial'].values

    res_group = minimize(
        neg_log_likelihood,
        x0=[0.5, 3.0, 0.0, 0.0],
        args=(x_data_group, n_trials_group, y_data_group),
        bounds=[(0.01, 1.0), (0.1, 10.0), (0.0, 1), (0.0, 1)],
        method='powell'
    )


    if res_group.success:
        alpha_group, beta_group, gamma_group, lambd_group = res_group.x
        y_group_fit = weibull_psychometric(x_fit, alpha_group, beta_group, gamma_group, lambd_group)


combined_fit_df = pd.concat(all_fit_results, ignore_index=True)

threshold_df = pd.DataFrame(thresholds_50)
threshold_df_tap = threshold_df.dropna(subset = ['threshold_percentage'])
threshold_df_tap['unitdur'] = threshold_df_tap['unitdur'].astype(str)  # Convert to string for categorical plotting

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
    data=threshold_df_tap,
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
    data=threshold_df_tap,
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
medians = threshold_df_tap.groupby('unitdur')['threshold_percentage'].median()
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
fig.savefig('threshold_violin_sms.svg', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
plt.show()


# %%
percentage_group_data_pid = df_tap_clean.groupby(['percentage', 'participant_id']).agg(
    avg_sig=('significant', 'mean'),
    std=('significant', 'std'),
    n_sig=('significant', 'sum'),
    n_trial=('significant', 'count'),
).reset_index()



percentage_group_data = df_tap_clean.groupby(['percentage']).agg(
    avg_sig=('significant', 'mean'),
    std=('significant', 'std'),
    n_sig=('significant', 'sum'),
    n_trial=('significant', 'count'),
).reset_index()



x_fit = np.linspace(0, 1, 500)

x_data_group = percentage_group_data['percentage'].values
y_data_group = percentage_group_data['n_sig'].values
n_trials_group = percentage_group_data['n_trial'].values

# --- Fit Weibull to individual using MLE ---
init_params = [0.5, 3.0, 0.17, 0]  # alpha, beta, gamma, lambd
bounds = [(0.01, 1), (0.01, 50), (0, 1), (0, 1)]

res = minimize(neg_log_likelihood, init_params,
    args=(x_data_group, n_trials_group, y_data_group),
    bounds=bounds, method='Powell')

if res.success:
    alpha_fit, beta_fit, gamma_fit, lambd_fit = res.x

    fit_results.append({
        'unit_dur': unitdur,
        'alpha': alpha_fit,
        'beta': beta_fit,
        'gamma': gamma_fit,
        'lambda': lambd_fit
    })

    y_fit = weibull_psychometric(x_fit, alpha_fit, beta_fit, gamma_fit, lambd_fit)



unit_durations = [0.4, 0.7, 1.0]
colors = ['blue', 'green', 'red']
x_fit = np.linspace(0, 1, 500)


all_fit_results = []
for unitdur in unit_durations:

    group_data = summary_all[summary_all['unit_dur'] == unitdur]
    x_data_group = group_data['percentage'].values
    y_data_group = group_data['n_sig'].values
    n_trials_group = group_data['n_trial'].values

    # --- Initial parameters and bounds ---
    if unitdur == 0.4:
        init_params = [0.5, 3.0, 0.2, 0.05]  # last param = sigma
        bounds = [(0.01, 1), (0.01, 20), (0, 0.5), (0, 0.5)]
    elif unitdur == 0.7:
        init_params = [0.5, 3.0, 0.1, 0]
        bounds = [(0.01, 1), (0.01, 20), (0, 0.5), (0, 0.5)]
    elif unitdur == 1.0:
        init_params = [0.5, 3.0, 0.2, 0.15]
        bounds = [(0.01, 1), (0.01, 20), (0, 0.5), (0, 0.5)]


    # --- Fit Weibull to individual using MLE ---
    res = minimize(neg_log_likelihood, init_params,
            args=(x_data_group, n_trials_group, y_data_group),
            bounds=bounds, method='Powell')

    if res.success:
        alpha_fit, beta_fit, gamma_fit, lambd_fit = res.x

        fit_results.append({
            'unit_dur': unitdur,
            'alpha': alpha_fit,
            'beta': beta_fit,
            'gamma': gamma_fit,
            'lambda': lambd_fit
        })

        y_fit = weibull_psychometric(x_fit, alpha_fit, beta_fit, gamma_fit, lambd_fit)

    


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
y_data_group = percentage_group_data['n_sig'].values
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
    .agg(avg=('avg_sig', 'mean'),
         std=('avg_sig', 'std'),
         n_participants=('avg_sig', 'count'))
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
    group_data = summary_all[summary_all['unit_dur'] == unitdur]
    x_data_group = group_data['percentage'].values
    y_data_group = group_data['n_sig'].values
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
        summary_participant[summary_participant['unit_dur'] == unitdur]   # unit_dur, not unitdur
        .groupby(['percentage'])
        .agg(avg_sig=('mean_sig', 'mean'),
            std_sig=('mean_sig', 'std'),
            n_participants=('mean_sig', 'count'))
        .reset_index()
    )
    grouped['sem_sig'] = grouped['std_sig'] / np.sqrt(grouped['n_participants'])

    ax_sep.plot(grouped['percentage'], grouped['avg_sig'], 'o',
                color=color, markersize=3)
    ax_sep.fill_between(grouped['percentage'],
                        grouped['avg_sig'] - grouped['sem_sig'],
                        grouped['avg_sig'] + grouped['sem_sig'],
                        color=color, alpha=0.3)

ax_sep.axhline(y=0.5, color='black', linestyle='--', linewidth=1, alpha = 0.7)
ax_sep.legend(title='', fontsize=5)
ax_sep.set_xticks([0, 0.5, 1])
ax_sep.tick_params(axis='both', labelsize=15)

plt.tight_layout()
fig.savefig('weibull_combined_sms.svg', dpi=300,
            transparent=True, bbox_inches='tight', pad_inches=0)
plt.show()



# %%
# data frame for R analysis
prop_sig_df = df_tap_clean.groupby(['unit_dur', 'percentage', 'participant_id']).agg(
    mean_sig=('significant', 'mean'),
    std_sig=('significant', 'std'),
    n_sig = ('significant', 'sum'),
    n_trial = ('significant', 'count'),
).reset_index()

prop_sig_df.to_csv('prop_sig_df_for_r.csv', index=False)

# %%
sig_df = df_tap_clean[df_tap_clean['significant'] == 1]
# all delays
# Prepare the result list for storing delays
tap_onset_delays_list = []
# Iterate over each row in the DataFrame
for index, row in sig_df.iterrows():
    # Select the last 20 presentation cycles (actual_onset_s)
    actual_window = row['actual_onset_s']
    
    # Select tap onsets (tap_onset_s) that fall within the range of the last 20 presentation cycles
    # Define the range as the first and last actual onset in the window
    start_range = actual_window[0]
    # till the end of the trial which includes: time point of presentation, the duration of the presentation, zero pad
    zero_pad = 0.2
    # end_range = actual_window[-1] + row['unit_dur'] + zero_pad
    end_range = actual_window[-1]
    
    # Select tap onsets that are within this range
    tap_window = np.array(row['tap_onset_s'])
    valid_tap_onsets = tap_window[(tap_window >= start_range) & (tap_window <= end_range)]

    # Find the closest indices in actual_window that are less than or equal to the tap onsets using np.searchsorted
    closest_indices = np.searchsorted(actual_window, valid_tap_onsets, side='right') - 1
    
    # Ensure indices are valid (i.e., actual onset is before tap onset)
    valid_indices = (closest_indices >= 0)
    closest_actuals = np.array(actual_window)[closest_indices[valid_indices]]
    valid_tap_onsets = valid_tap_onsets[valid_indices]
    
    # Compute tap onset delays (difference between tap and actual)
    tap_onset_delays = valid_tap_onsets - closest_actuals
    
    # Add the computed delays for this row to the list
    tap_onset_delays_list.append(tap_onset_delays)

# After processing all rows, you can store the delays in the DataFrame or another structure
sig_df['tap_onset_delays_all'] = tap_onset_delays_list


sig_df[['tapping_phases_all', 'tapping_vectors_all']] = sig_df.apply(
    lambda row: pd.Series(get_tapping_vectors(row['tap_onset_delays_all'], row['unit_dur'])), 
    axis=1
)


from scipy.stats import circstd, circmean
def locate_within_range_taps(radians_all, radians_last, consec_len = 5):
    # Mean and standard deviation of the delays (in radians)
    mean_delay = circmean(radians_last)
    std_delay = circstd(radians_last)

    # Compute circular range limits
    lower_bound = (mean_delay - std_delay) % (2 * np.pi)
    upper_bound = (mean_delay + std_delay) % (2 * np.pi)

    # Initialize a list to store indices of values outside the range
    outside_indices = []

    for i, angle in enumerate(radians_all):
        # Normalize angle to [0, 2π]
        norm_angle = angle % (2 * np.pi)
        if not (
            lower_bound <= norm_angle <= upper_bound
            if lower_bound < upper_bound
            else norm_angle >= lower_bound or norm_angle <= upper_bound
        ):
            outside_indices.append(i)

    all_indices = set(range(len(radians_all)))
    inside_indices = sorted(list(all_indices - set(outside_indices)))
    n_outside = len(outside_indices)
    min_inside_index = min(inside_indices) if inside_indices else None

    # --- Find the first inside index followed by `consec_len` consecutive inside values ---
    min_stable_index = None
    for idx in inside_indices:
        # Check if the next (consec_len - 1) indices are also in inside_indices
        window = list(range(idx, idx + consec_len))
        if all(i in inside_indices for i in window):
            min_stable_index = idx
            break

    # Return everything including bounds
    return outside_indices, n_outside, min_inside_index, inside_indices, lower_bound, upper_bound, min_stable_index


# Apply function to each row and extract results into separate columns
results_df = sig_df.apply(
    lambda row: pd.Series(
        locate_within_range_taps(row['tapping_phases_all'], row['tapping_phases_stabil'], consec_len=5)
    ), axis=1
)

results_df.columns = [
    'outside_indices', 'n_outside', 'min_inside_index', 'inside_indices', 'lower_bound', 'upper_bound', 'min_stable_index'
]

# Merge results back into the original dataframe
sig_df = pd.concat([sig_df, results_df], axis=1)


# Ensure fresh phase range calc
sig_df['phase_range'] = (sig_df['upper_bound'] - sig_df['lower_bound']) % (2 * np.pi)

# Filter out missing values
valid_data = sig_df[['participant_id', 'percentage', 'unit_dur', 'min_inside_index', 'phase_range']].dropna()


grouped_participant = valid_data.groupby(['percentage', 'unit_dur', 'participant_id']).agg({
    'min_inside_index': 'median',
    'phase_range': 'mean'
}).reset_index()


filtered_tap_data = grouped_participant[grouped_participant['percentage'] == 1]
filtered_tap_data['min_inside_index'] = filtered_tap_data['min_inside_index'] + 1



# %%
#=====================
# VIOLIN PLOT OPTION FOR MIN INDEX
#=====================
colors = {0.4: "blue", 0.7: "green", 1.0: "red"}
colors_str = {str(k): v for k, v in colors.items()}

# Turn off Seaborn grid styling
sns.set_style("white")
plt.rcParams['font.family'] = 'Arial'
fig, ax = plt.subplots(figsize=(2.7, 2.2))

# Violin plot — distribution shape
sns.violinplot(
    data=filtered_tap_data,
    x='unit_dur',
    y='min_inside_index',
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
    data=filtered_tap_data,
    x='unit_dur',
    y='min_inside_index',
    color='black',
    alpha=0.4,
    size=3,
    jitter=0.12,                 # horizontal jitter (0 = stacked, ~0.2 = wide spread)
    ax=ax,
    edgecolor='white',
    linewidth=0.8,
)

# # Optional: add a median line across each violin
medians = filtered_tap_data.groupby('unit_dur')['min_inside_index'].median()
for i, (group_name, median_val) in enumerate(medians.items()):
    ax.hlines(median_val, i - 0.25, i + 0.25, color='black', linewidth=2, zorder=10, alpha = 0.6)

# Labels & style
ax.set_xlabel('')
ax.set_ylabel('')
ax.grid(False)
plt.xticks(fontsize=15)
plt.yticks([2.5, 7.5, 12.5], fontsize=15)
# plt.ylim([0, 1])
plt.tight_layout()
fig.savefig('min_index_violin_sms.svg', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
plt.show()


# %%
filtered_sig_df = sig_df[sig_df['percentage'] == 1]

# %%
# Pick a few trials to visualize (e.g., first 2 rows)
example_trials = filtered_sig_df.iloc[33:34]
plt.rcParams['font.family'] = 'Arial'
for idx, row in example_trials.iterrows():
    tap_delays = row['tap_onset_delays_all']   
    n_taps = len(tap_delays)
    tap_indices = np.arange(1, n_taps + 1)

    # Convert tap delays (sec) → radians in 0–2π
    tap_delays_rad = (tap_delays / row['unit_dur']) * 2*np.pi
    
    # Convert phase range to radians
    phase_range_rad = row['phase_range']

    fig, ax = plt.subplots(figsize=(2.2, 0.9))

    ax.plot(tap_indices, tap_delays_rad, 'o-', color='#0343DF', linewidth=1, markersize=2)

    # Shade mean ± phase range
    mean_delay_rad = np.mean(tap_delays_rad)
    ax.fill_between(
        tap_indices,
        mean_delay_rad - phase_range_rad,
        mean_delay_rad + phase_range_rad,
        color='gray', alpha=0.2
    )

    # ax.set_xlabel("Tap Index", fontsize=70)
    # ax.set_ylabel("Phase \n(rad)", fontsize=70)
    ax.set_ylim(0, 2*np.pi)

    y_min = 0 - 0.1*np.pi
    y_max = 2*np.pi + 0.1*np.pi
    ax.set_ylim(y_min, y_max)

    # Set y-ticks at 0, π/2, π, 3π/2, 2π
    y_ticks = [0, 2*np.pi]
    y_labels = [r"$0$",  r"$2\pi$"]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=12)
    x_ticks = [0, 10, 20, 30]
    ax.set_xticks(x_ticks)

    ax.tick_params(labelsize=12)
    fig.savefig('good_trial_example.svg', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.show()

# %%
