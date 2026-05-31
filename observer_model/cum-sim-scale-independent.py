# %%
import numpy as np
import pandas as pd
import itertools as it
from stimgen import make_julez_stream, rms
from scipy import signal as sps
from statsmodels.tsa.stattools import acf


# %%
# REPRODUCIBILITY
# Single seed at the top — applied via np.random.seed.
# Change the integer to get a different (still-reproducible) realization.
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# %%
# SIMULATION MODE
# Two modes:
#   "single_base": one fixed base criterion + trial-level Gaussian criterion noise.
#                  Recommended primary mode.
#   "sweep_base":  sweeps over a range of base criteria. Trial-level noise is
#                  still applied on top, so both sources of variability combine.
SIMULATION_MODE = "sweep_base"   # "single_base" or "sweep_base"


# PARAMETERS
period_min, period_max = 20,40
period_domain = np.logspace(np.log10(period_min), np.log10(period_max), num=3, dtype=int)
# period_domain = np.linspace(period_min, period_max, num=10, dtype=int)
snr_domain = np.linspace(0.0, 1.0, num=30).round(2)   # repetition coherence
internal_noise_level = 0.15
max_reps = 15


# Criterion model parameters
BASE_CRITERION = 0.3      # used in "single_base" mode
SIGMA_CRIT = 0.07          # trial-level Gaussian std on the criterion


# Baseline sweep range — used only in "sweep_base" mode.
# Narrowed from the original [0.1, 0.9] so that the longest period's
# scaled criterion remains within the reachable range of the ACF peak.
decision_domain = np.linspace(0.3, 0.4, num=10)

# Number of trials per cell.
# In single_base mode each cell is sampled num_trials times.
# In sweep_base mode each (cell, base) pair is sampled num_trials times,
# so total samples per cell = num_trials * len(decision_domain).
if SIMULATION_MODE == "sweep_base":
    num_trials = 50
else:
    num_trials = 100


# Build the iteration list of base criteria according to mode
if SIMULATION_MODE == "single_base":
    base_iter = [BASE_CRITERION]
elif SIMULATION_MODE == "sweep_base":
    base_iter = list(decision_domain)
else:
    raise ValueError(f"Unknown SIMULATION_MODE: {SIMULATION_MODE}")


results = []
# SIMULATION LOOP
for trial in range(num_trials):

    print(f"TRIAL: {trial}/{num_trials}")

    for period, snr, base_thresh in it.product(period_domain, snr_domain, base_iter):

        # Trial-level criterion noise — sampled ONCE and held across the rep loop.
        # This represents the participant having a stable but noisy criterion
        # for the duration of one trial, not a fresh criterion at each cycle.
        if SIGMA_CRIT > 0:
            criterion_noise = np.random.normal(0, SIGMA_CRIT)
        else:
            criterion_noise = 0.0
        decision_thresh = base_thresh + criterion_noise
        decision_thresh = float(np.clip(decision_thresh, 0.0, 1.0))

        detected = False
        max_corr_at_last = 0

        # tries 1 to 30 repetitions to see at what point a repetition becomes detectable
        for reps in range(1, max_reps + 1):
            # Generate stimulus
            # synthesizes a repeating pattern of reps cycles with specified SNR and unit period
            stim = make_julez_stream(snr=snr, len_unit=period, num_reps=reps)

            # Add internal noise
            int_noise = np.random.uniform(-1, 1, size=stim.shape)
            int_noise -= int_noise.mean()
            int_noise /= rms(int_noise)
            stim_noised = stim + internal_noise_level * int_noise
            stim_noised -= stim_noised.mean()
            stim_noised /= rms(stim_noised)

            # Compute autocorrelation
            corr, confint = acf(stim_noised, adjusted=False, alpha=0.05, nlags=stim_noised.shape[0]-1)

            # Peak selection (unified, correct version)
            peak_idx, _ = sps.find_peaks(corr, height=0)
            if peak_idx.size == 0:
                peak_idx = np.array([np.random.randint(0, corr.size)])
            best_idx = peak_idx[np.argmax(corr[peak_idx])]

            max_corr = corr[best_idx]
            max_confint = confint[best_idx]
            size_confint = max_confint[1] - max_confint[0]

            max_corr_at_last = max_corr

            # Check detection against threshold
            if max_corr >= decision_thresh:
                results.append({
                    'Period': period,
                    'SNR': snr,
                    'BaseCriterion': base_thresh,
                    'Threshold': decision_thresh,
                    'Trial': trial,
                    'N_Cycles': reps,
                    'Detected': True,
                    'MaxCorr': max_corr
                })
                detected = True
                break  # stop early on detection

        # Move this outside the `reps` loop
        if not detected:
            results.append({
                'Period': period,
                'SNR': snr,
                'BaseCriterion': base_thresh,
                'Threshold': decision_thresh,
                'Trial': trial,
                'N_Cycles': max_reps,
                'Detected': False,
                'MaxCorr': max_corr_at_last
            })


# Convert to DataFrame
df_detect = pd.DataFrame(results)
# save the data frame

# Mode-specific filename so different runs don't clobber each other
out_path = f"ncycle_sim_results_scale_independent_seed{RANDOM_SEED}.csv"
df_detect.to_csv(out_path, index=False)