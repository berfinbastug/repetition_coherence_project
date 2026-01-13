# %%
import numpy as np
from stimgen import make_julez_stream, rms
from scipy import signal as sps
from matplotlib import pyplot as plt
import itertools as it
import pandas as pd
from statsmodels.tsa.stattools import acf
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase
from mpl_toolkits.axes_grid1 import make_axes_locatable


# %%
# CONTENTS:
# An ideal observer for the repetition detection/tapping task
# Simulations with various parameters
# Plotting of the results for each condition
# UNIT DURATION AGNOSTIC
# NOTHING CHANGES WITH PERIOD LENGTH
# by the way, SNR is not a good description because it is not SNR
# %%    
# SIMULATION PARAMETERS
title = "norm-corr"
period_min, period_max = 20, 40
period_domain = np.logspace(np.log10(period_min), np.log10(period_max), num=3, dtype=int)
reps_min, reps_max = 2, 31
# logarithmically spaced repetitions domain from 2 to 30
repetitions_domain = np.logspace(
    np.log10(reps_min), np.log10(reps_max), 
    num=10)
# Convert to int and remove duplicates
repetitions_domain = np.unique(repetitions_domain.astype(int))
snr_domain = np.linspace(0.0, 1.0, num=30).round(2)
# snr_domain = (1 - np.logspace(-1, 0, num=8)).round(2)
do_repeat_domain = [True] # [True, False]
decision_domain = np.linspace(0.2, 0.9, num=10)
with_statmodels = True # compute correlation with package statmodels
n_minus_k_norm = False  # normalize the correlation by (n-k) with statmodels
do_abs_corr = False # take the absolute value of autocorrelation function
plot_while_simulating = False # shows a plot for every trial
internal_noise_level = 0.2

# SIMULATION LOOP
# Initialize results lists
estimations = []
repeat_condition = []
period_condition = []
snr_condition = []
repetitions_condition = []
decision_threshold = []
errors = []
decisions = []
accuracies = []
max_autocorrs = []
sizes_confint = []

# create a loop with one period value but with different repetitions
# and snr values
num_simulations = 150
for n in range(0, num_simulations):

    print("SIMULATION NUMBER: ", n)
    
    for period, repetitions, snr, do_repeat, decision_thresh in it.product(  # experiment factors
        period_domain, repetitions_domain, snr_domain, do_repeat_domain, decision_domain):

        # Generate stimulus for observer
        stimulus = make_julez_stream(snr=snr, len_unit=period, num_reps=repetitions)

        # INTERNAL NOISE
        int_noise = np.random.uniform(-1.0, 1.0, size=[stimulus.shape[0]])
        int_noise -= int_noise.mean()
        int_noise /= rms(int_noise)

        noised_stim = stimulus + (internal_noise_level * int_noise)
        noised_stim -= noised_stim.mean()
        noised_stim /= rms(noised_stim)

        # PERCEPTION
        # Autocorrelation
        lags = sps.correlation_lags(noised_stim.shape[0], noised_stim.shape[0], mode="full")
        zero_lag_index = np.where(lags == 0)[0].item() 
        lags_nonneg = lags[zero_lag_index:]  # keep only positive lags
        
        if with_statmodels:
            # Autocorrelation with statsmodels package ('adjusted': normalized by n-k)
            corr, confint = acf(noised_stim, adjusted=n_minus_k_norm, alpha=.05, nlags=noised_stim.shape[0]-1)
        else:
            corr = sps.correlate(noised_stim, noised_stim, mode="full")
            corr = corr[zero_lag_index:] # keep only nonegative-lag correlations

        corr = np.abs(corr) if do_abs_corr else corr


        # ESTIMATION of periodicity lag
        # array of positive lag indices  sorted (min to max) according to corrrelation

        if with_statmodels:
            peak_idx, _ = sps.find_peaks(corr, height=0)
            # choose random index if no peaks are detected
            peak_idx = np.array([np.random.randint(0, corr.size)]) if peak_idx.size == 0 else peak_idx
            # Pick the highest peak
            best_idx = peak_idx[np.argmax(corr[peak_idx])]

        else:
            # stable = true, doesn't work in my computer
            # below is my solution to fede's single line code...
            indices = np.arange(1, len(corr)) # Create indices from 1 to len(corr)-1
            data_with_index = list(zip(corr[1:], indices)) # Combine correlation values with indices 
            sorted_data = sorted(data_with_index, key=lambda x: x[0])
            sortcorr_idx = np.array([x[1] for x in sorted_data])
            # sortcorr_idx = np.argsort(corr[1:], stable=True) + 1
            best_idx = sortcorr_idx[-1]
        
        best_lag = lags_nonneg[best_idx]
        estimation = (best_lag % period) + period  # best estimate of period

        # ERROR in estimation
        error = min(np.abs(estimation - period), estimation) # take the minimum due to circularity

        # DECISION
        max_corr = corr[best_idx]
        max_confint = confint[best_idx]
        size_confint = max_confint[1] - max_confint[0]
        decision = True if max_corr >= decision_thresh else False
        accuracy = 1 if decision == repeat_condition else 0

        # Aggregate simulation results
        estimations.append(estimation.item())  # lags
        repeat_condition.append(do_repeat)
        period_condition.append(period.item())
        snr_condition.append(snr.item())
        repetitions_condition.append(repetitions.item())
        decision_threshold.append(decision_thresh.item())
        errors.append(error.item())  # lag estimation error
        decisions.append(decision)
        accuracies.append(accuracy)
        max_autocorrs.append(max_corr.item())
        sizes_confint.append(size_confint.item())



# %%
# BUILD DATAFRAME
col_names = ['Estimation', 'Repeated', 'Period', "SNR", 
             "Reps", "Threshold", "Error", "Decision", "Accuracy", "Autocorr", "SizeConf"]

df_wide = pd.DataFrame(
    list(zip(estimations, repeat_condition, period_condition, 
            snr_condition, repetitions_condition, decision_threshold, errors, 
            decisions, accuracies, max_autocorrs, sizes_confint)), 
    columns=col_names)
df_wide["id"] = df_wide.index
df_wide["estimation_accuracy"] = df_wide["Period"] == df_wide["Estimation"]
df = df_wide.set_index(["id", "Period", "SNR", "Reps", 
                        "Decision", "Repeated"]) # to long format

# save the dataframe
# df_wide.to_csv("process_model_simulations_scale_independent_new_params.csv", index=False)

# %%
# read the data frame
df_wide = pd.read_csv("process_model_simulations_scale_independent_new_params.csv")    


# %%
df_wide["Decision"] = df_wide["Decision"].astype(int)
# Compute average accuracy for each combination of SNR and Period
df_avg = df_wide.groupby(["SNR", "Period"])["Decision"].mean().reset_index()

# %%
# Define continuous colormap from blue to green to red
# cmap = mcolors.LinearSegmentedColormap.from_list('blue_green_red', ['#0984ea', '#23d619', '#ad0716'])
cmap = mcolors.LinearSegmentedColormap.from_list('blue_green_red', ['#069AF3', '#006400', '#DC143C'])

# Unique sorted Periods and normalization
periods = np.sort(df_avg['Period'].unique())
norm = mcolors.Normalize(vmin=periods.min(), vmax=periods.max())

fig, ax = plt.subplots(figsize=(10, 8))

# Plot each period line with corresponding color
for period in periods:
    sub_df = df_avg[df_avg['Period'] == period]
    color = cmap(norm(period))
    ax.plot(sub_df['SNR'], sub_df['Decision'], color=color, linewidth=9, marker='o', markersize = 15)

# ax.set_xlabel('Repetition\nCoherence', fontsize=60)
# ax.set_ylabel('P(Yes)', fontsize=70)
ax.grid(False)
ax.tick_params(axis='both', which='major', labelsize=70)
ax.set_yticks([0.2, 0.5, 0.8])

# Add colorbar without ticks or label
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)

cb = ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
cb.set_ticks([])         # Remove ticks
cb.set_ticklabels([])    # Remove tick labels
cb.set_label('')         # Remove label text

plt.show()



# %%
# --- Ensure Autocorr is numeric ---
df_wide["Autocorr"] = pd.to_numeric(df_wide["Autocorr"], errors='coerce')

# --- Determine the lowest, middle, and highest Periods ---
unique_periods = sorted(df_wide["Period"].unique())
low_p, mid_p, high_p = unique_periods[0], unique_periods[len(unique_periods)//2], unique_periods[-1]
selected_periods = [low_p, mid_p, high_p]


# --- Base colors per period ---
base_colors = {
    low_p: "#069AF3",   # blue
    mid_p: "#006400",   # green
    high_p: "#DC143C"   # red
}

# --- Prepare 10 equally spaced SNR values ---
all_snrs = sorted(df_wide["SNR"].unique())
num_snr_levels = 10
selected_indices = np.round(np.linspace(0, len(all_snrs) - 1, num_snr_levels)).astype(int)
snrs = [all_snrs[i] for i in selected_indices]

snr_min, snr_max = min(snrs), max(snrs)
snr_to_index = {snr: i for i, snr in enumerate(snrs)}
norm = mcolors.Normalize(vmin=snr_min, vmax=snr_max)

# --- Function to create custom colormap ---
def get_colormap(base_hex, n_levels):
    rgb = np.array(mcolors.to_rgb(base_hex))
    white = np.array([1, 1, 1])
    colors = [mcolors.to_hex((1 - t) * white + t * rgb) for t in np.linspace(0.0, 1.0, n_levels)]
    return colors  # darkest = highest SNR

# --- Set up figure and axes ---
fig, axs = plt.subplots(1, 3, figsize=(25, 8), sharey=True)
# fig.subplots_adjust(wspace=0.3, right=0.88, bottom=0.)

# --- Plotting ---
for i, period in enumerate(selected_periods):
    ax = axs[i]
    sub_df = df_wide[df_wide["Period"] == period]
    colormap = get_colormap(base_colors[period], len(snrs))

    for snr in snrs:
        df_plot = sub_df[sub_df["SNR"] == snr].groupby("Reps")["Autocorr"].mean().reset_index()
        color = colormap[snr_to_index[snr]]
        ax.plot(df_plot["Reps"], df_plot["Autocorr"], color=color, linewidth=9)
        ax.scatter(df_plot["Reps"], df_plot["Autocorr"],
                   color=color, s=60, edgecolor='black', linewidth=5, alpha=0.8)

    ax.set_xticks([5, 15, 30])
    ax.set_yticks([0.2, 0.5, 0.8])
    ax.tick_params(axis='both', labelsize=70)

    # if i == 0:
    #     ax.set_ylabel("Auto Corr", fontsize=70)

    ax.grid(False)

    # --- Custom colorbar ---
    box = ax.get_position()
    bar_width = 0.015
    bar_pad = 0.01
    cbar_ax = fig.add_axes([
        box.x1 + bar_pad,
        box.y0,
        bar_width,
        box.height
    ])
    cmap = mcolors.LinearSegmentedColormap.from_list("custom", get_colormap(base_colors[period], 256))
    cb = ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='vertical')

    if i == len(selected_periods) - 1:
        # cb.set_label("Coherence", fontsize=40)
        cb.ax.tick_params(labelsize=40)
        cb.ax.set_yticks([0, 0.5, 1])
    else:
        cb.ax.set_yticks([])
        cb.ax.set_ylabel("")

# # --- Shared X-axis label ---
# fig.text(0.5, 0.08, "N Repetition", ha='center', va='center', fontsize=70)

plt.show()


