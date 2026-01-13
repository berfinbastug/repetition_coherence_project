# %%
import numpy as np
import pandas as pd
import itertools as it
from stimgen import make_julez_stream, rms
from scipy import signal as sps
from statsmodels.tsa.stattools import acf
from scipy.stats import sem
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors



# %%
# PARAMETERS
period_min, period_max = 20,40
period_domain = np.logspace(np.log10(period_min), np.log10(period_max), num=3, dtype=int)
# period_domain = np.linspace(period_min, period_max, num=10, dtype=int)
snr_domain = np.linspace(0.0, 1.0, num=30).round(2)   # repetition coherence
decision_domain = np.linspace(0.2, 0.9, 10)   # range of thresholds
internal_noise_level = 0.2
max_reps = 30
num_trials = 150

results = []


# SIMULATION LOOP
for trial in range(num_trials):
    print(trial)
    for period, snr, decision_thresh in it.product(period_domain, snr_domain, decision_domain):

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
            peak_idx, _ = sps.find_peaks(corr, height=0)
            
            # choose random index if no peaks are detected
            peak_idx = np.array([np.random.randint(0, corr.size)]) if peak_idx.size == 0 else peak_idx
            
            # Pick the highest peak
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
                'Threshold': decision_thresh,
                'Trial': trial,
                'N_Cycles': max_reps,
                'Detected': False,
                'MaxCorr': max_corr_at_last
            })


# Convert to DataFrame
df_detect = pd.DataFrame(results)
# save the data frame
df_detect.to_csv('ncycle_sim_results_scale_independent_new_params.csv', index=False)

#%%
# read the data frame
df_detect = pd.read_csv('ncycle_sim_results_scale_independent_new_params.csv')

# read the data csv file
period_min, period_max = 20,40
period_domain = np.logspace(np.log10(period_min), np.log10(period_max), num=3, dtype=int)
# %%
# Set up color map
cmap = mcolors.LinearSegmentedColormap.from_list(
    'blue_green_red',
    ['#069AF3', '#006400', '#DC143C'],
    N=256  # Smoother gradient
)


# Normalize periods for color mapping
norm = plt.Normalize(vmin=period_domain.min(), vmax=period_domain.max())
period_colors = {period: cmap(norm(period)) for period in period_domain}

# Aggregate across all thresholds
summary = df_detect.groupby(['SNR', 'Period']).agg(
    mean_ncycle=('N_Cycles', 'mean'),
    sem_ncycle=('N_Cycles', sem),
    mean_detected=('Detected', 'mean')
).reset_index()

# Fill missing detection values to avoid NaNs
summary['mean_detected'] = summary['mean_detected'].fillna(0)

# Normalize accuracy for transparency
min_detect = summary['mean_detected'].min()
max_detect = summary['mean_detected'].max()
range_detect = max_detect - min_detect if max_detect > min_detect else 1  # avoid divide-by-zero
summary['correct_norm'] = ((summary['mean_detected'] - min_detect) / range_detect).fillna(0)

# %%
# Plotting
fig, ax = plt.subplots(figsize=(10, 8))

for period in period_domain:
    sub = summary[summary['Period'] == period].sort_values('SNR')
    if sub.empty:
        continue

    x = sub['SNR']
    y = sub['mean_ncycle']
    yerr = sub['sem_ncycle']
    color = period_colors[period]

    # SEM band
    ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.2)

    # Line segments with alpha = detection rate
    for i in range(len(sub) - 1):
        x_pair = x.iloc[i:i + 2]
        y_pair = y.iloc[i:i + 2]
        alpha = float(np.clip(sub['correct_norm'].iloc[i], 0, 1))
        ax.plot(x_pair, y_pair, color=color, alpha=alpha, linewidth=13)

# Axis labels and formatting
# ax.set_xlabel("Repetition\nCoherence", fontsize=60)
# ax.set_ylabel("N Cycles", fontsize=70)
ax.set_yticks([5, 15, 25])
ax.tick_params(axis='both', which='major', labelsize=70)
# plt.tight_layout()
plt.show()

