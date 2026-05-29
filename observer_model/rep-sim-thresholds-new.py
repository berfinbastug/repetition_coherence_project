# %%
import numpy as np
from stimgen import make_julez_stream, rms
from scipy import signal as sps
import itertools as it
import pandas as pd
from statsmodels.tsa.stattools import acf
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase
from mpl_toolkits.axes_grid1 import make_axes_locatable

# %%
# REPRODUCIBILITY
# A single seed at the top makes the entire simulation deterministic.
# Change this number to get a different (but still reproducible) realization.
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# %%
# SIMULATION MODE
# Two modes:
#   "single_base": one fixed base criterion + trial-level Gaussian criterion noise.
#                  This is the recommended primary mode.
#   "sweep_base":  the original behavior — sweep over a range of base criteria,
#                  aggregating across them. Used as a robustness check.
#
# In "single_base" mode, decision_domain is ignored.
# In "sweep_base" mode, sigma_crit can be set to 0 to recover the original
# block-only variability, or kept positive to combine both sources.
SIMULATION_MODE = "sweep_base"   # "single_base" or "sweep_base"




# %%
# SIMULATION PARAMETERS
title = "norm-corr"
period_min, period_max = 20, 40
period_domain = np.logspace(np.log10(period_min), np.log10(period_max),
                            num=3, dtype=int)
# period_domain = np.linspace(period_min, period_max, num = 3, dtype = int)
reps_min, reps_max = 2, 31
repetitions_domain = np.logspace(np.log10(reps_min), np.log10(reps_max),
                                  num=10)
repetitions_domain = np.unique(repetitions_domain.astype(int))
snr_domain = np.linspace(0.0, 1.0, num=30).round(2)

do_repeat_domain = [True]

n_minus_k_norm = False
internal_noise_level = 0.15

# Criterion model parameters
BASE_CRITERION = 0.3      # used in "single_base" mode
SIGMA_CRIT = 0.07          # trial-level Gaussian noise on the criterion (both modes)
CRITERION_SLOPE = 0.3     # how much criterion rises with normalized log-period

# Baseline sweep range — used only in "sweep_base" mode
decision_domain = np.linspace(0.3, 0.40, num=10)

# Number of simulations per cell.
# In "single_base" mode each cell is sampled num_simulations times.
# In "sweep_base" mode each (cell, base_threshold) pair is sampled num_simulations times,
# so total samples per cell is num_simulations * len(decision_domain).
# To keep total trial counts comparable across modes, we use a higher value
# in single_base.
if SIMULATION_MODE == "single_base":
    num_simulations = 500
else:  # sweep_base
    num_simulations = 50

# %%
# SIMULATION LOOP
estimations = []
repeat_condition = []
period_condition = []
snr_condition = []
repetitions_condition = []
base_criterion_used = []      # the trial's base criterion (before period scaling)
effective_threshold_used = []  # the actual threshold applied on this trial
errors = []
decisions = []
accuracies = []
max_autocorrs = []
sizes_confint = []

# Build the iteration list according to mode
if SIMULATION_MODE == "single_base":
    base_iter = [BASE_CRITERION]
elif SIMULATION_MODE == "sweep_base":
    base_iter = list(decision_domain)
else:
    raise ValueError(f"Unknown SIMULATION_MODE: {SIMULATION_MODE}")

for n in range(num_simulations):
    if n % 50 == 0:
        print(f"SIMULATION NUMBER: {n}/{num_simulations}")

    for period, repetitions, snr, do_repeat, base_thresh in it.product(
        period_domain, repetitions_domain, snr_domain,
        do_repeat_domain, base_iter):

        # Period-dependent decision criterion
        # Logarithmic scaling: period_min -> 0, period_max -> 1
        raw_scaling = np.log(period / period_min)
        max_raw_scaling = np.log(period_max / period_min)
        normalized_scaling = raw_scaling / max_raw_scaling

        # Apply criterion slope
        scaled_criterion = base_thresh + (normalized_scaling * CRITERION_SLOPE)

        # Add trial-level Gaussian noise to the criterion (the new piece)
        if SIGMA_CRIT > 0:
            criterion_noise = np.random.normal(0, SIGMA_CRIT)
        else:
            criterion_noise = 0.0
        decision_thresh = scaled_criterion + criterion_noise

        # Clip to [0, 1] — outside this range the criterion is meaningless
        decision_thresh = float(np.clip(decision_thresh, 0.0, 1.0))

        # Generate stimulus
        stimulus = make_julez_stream(snr=snr, len_unit=period,
                                     num_reps=repetitions)

        # Internal sensory noise
        int_noise = np.random.uniform(-1.0, 1.0, size=[stimulus.shape[0]])
        int_noise -= int_noise.mean()
        int_noise /= rms(int_noise)

        noised_stim = stimulus + (internal_noise_level * int_noise)
        noised_stim -= noised_stim.mean()
        noised_stim /= rms(noised_stim)

        # PERCEPTION: autocorrelation
        lags = sps.correlation_lags(noised_stim.shape[0],
                                     noised_stim.shape[0], mode="full")
        zero_lag_index = np.where(lags == 0)[0].item()
        lags_nonneg = lags[zero_lag_index:]

        corr, confint = acf(noised_stim, adjusted=n_minus_k_norm, alpha=.05,
                            nlags=noised_stim.shape[0]-1)

        # Peak selection (unified branch — no inconsistency)
        peak_idx, _ = sps.find_peaks(corr, height=0)
        if peak_idx.size == 0:
            peak_idx = np.array([np.random.randint(0, corr.size)])
        best_idx = peak_idx[np.argmax(corr[peak_idx])]

        best_lag = lags_nonneg[best_idx]
        estimation = (best_lag % period) + period

        # Estimation error (kept for completeness; note: with the
        # estimation formula above this is just |best_lag % period|)
        error = min(np.abs(estimation - period), estimation)

        # DECISION
        max_corr = corr[best_idx]
        max_confint = confint[best_idx]
        size_confint = max_confint[1] - max_confint[0]
        decision = bool(max_corr >= decision_thresh)
        accuracy = 1 if decision == do_repeat else 0

        # Aggregate
        estimations.append(estimation.item())
        repeat_condition.append(do_repeat)
        period_condition.append(period.item())
        snr_condition.append(snr.item())
        repetitions_condition.append(repetitions.item())
        base_criterion_used.append(float(base_thresh))
        effective_threshold_used.append(decision_thresh)
        errors.append(error.item())
        decisions.append(decision)
        accuracies.append(accuracy)
        max_autocorrs.append(max_corr.item())
        sizes_confint.append(size_confint.item())


# %%
# BUILD DATAFRAME
col_names = ['Estimation', 'Repeated', 'Period', "SNR",
             "Reps", "BaseCriterion", "Threshold", "Error",
             "Decision", "Accuracy", "Autocorr", "SizeConf"]

df_wide = pd.DataFrame(
    list(zip(estimations, repeat_condition, period_condition,
             snr_condition, repetitions_condition,
             base_criterion_used, effective_threshold_used,
             errors, decisions, accuracies,
             max_autocorrs, sizes_confint)),
    columns=col_names)

df_wide["id"] = df_wide.index
df_wide["estimation_accuracy"] = df_wide["Period"] == df_wide["Estimation"]

# Save with a mode-specific filename so you don't overwrite previous runs
out_path = f"acf_model_simulations_threshold_scaled_seed{RANDOM_SEED}.csv"
df_wide.to_csv(out_path, index=False)
