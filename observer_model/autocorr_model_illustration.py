# %%
import numpy as np
from stimgen import make_julez_stream, rms
from scipy import signal as sps
from matplotlib import pyplot as plt
import itertools as it
import pandas as pd
import seaborn as sns
from statsmodels.tsa.stattools import acf
from matplotlib import cm, colors as mcolors
from matplotlib.cm import Purples



def rms(signal):
    return np.sqrt(np.mean(np.abs(signal) ** 2))



# %%
rng = np.random.default_rng(seed=42)  # for reproducibility
len_unit = 20
num_reps = 6
snr = 0.8  # Chosen SNR level

# Generate unit signal and full repeated signal
sig_unit = rng.uniform(-1.0, 1.0, size=len_unit)
signal = np.tile(sig_unit, reps=num_reps)
signal -= signal.mean()
signal /= rms(signal)

# Generate noise
noise = rng.uniform(-1.0, 1.0, size=signal.shape[0])
noise -= noise.mean()
noise /= rms(noise)

# Mix signal and noise at chosen SNR
stim = snr * signal + (1 - snr) * noise
stim -= stim.mean()
stim /= rms(stim)

# %%
# PLOT ONLY THE INITIAL SIGNAL
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(signal[:20], label='Pattern (Clean Signal)', linewidth=9, linestyle='--', color='gray', alpha = 0.8)
ax.grid(False)

plt.tight_layout()
plt.show()


# %%# Plot composite signal
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(signal[0:40], 
        label='Pattern (Clean Signal)', 
        linewidth=8, 
        linestyle='--', 
        color="0.4")

ax.plot(noise[0:40], 
        label='Noise', 
        linewidth=8, 
        linestyle=':', 
        color="0.6")

ax.plot(stim[0:40], 
        label=f'Final Stimulus (SNR = {snr})', 
        linewidth=11, 
        color=Purples(0.8))

# draw a vertical line cutting through 20 to mark the middle of the signal
ax.axvline(x=len_unit, color='green', linewidth=11, alpha=0.8)
# ax.set_title("Stimulus Composition: Pattern + Noise", fontsize=16)
ax.set_xlabel("Samples", fontsize=80)
# make x axis ticks from 0, 20, and 40 only
ax.set_xticks([0, 20, 40])
# increase tick size to 30
ax.tick_params(axis='x', labelsize=80)
ax.set_ylabel("Amp.", fontsize=80)
ax.set_yticks([-2, 0, 2])
ax.tick_params(axis='y', labelsize=80)
# ax.legend(fontsize=12)
ax.grid(False)

# plt.tight_layout()
plt.show()


# %%
# Compute autocorrelation
# Autocorrelation with statsmodels package ('adjusted': normalized by n-k)
corr, confint = acf(stim, adjusted=False, alpha=.05, nlags=stim.shape[0]-1)
lags = sps.correlation_lags(stim.shape[0], stim.shape[0], mode="full")
zero_lag_index = np.where(lags == 0)[0].item() 
lags_nonneg = lags[zero_lag_index:]  # keep only positive lags
peak_idx, _ = sps.find_peaks(corr, height=0)
# choose random index if no peaks are detected
peak_idx = np.array([np.random.randint(0, corr.size)]) if peak_idx.size == 0 else peak_idx
# Pick the highest peak
best_idx = peak_idx[np.argmax(corr[peak_idx])]


# %%
max_val = np.max(corr[1:40])
print(max_val)
# Plot autocorrelation only
fig, ax = plt.subplots(figsize=(14, 6))
# Plot autocorrelation curve
ax.plot(lags_nonneg[1:40], corr[1:40], 'black', linewidth=9)
# Axes labels and formatting
ax.set_xlabel("Lag", fontsize=80)
ax.set_ylabel("Auto Corr", fontsize=70)
ax.axhline(0.6, color='red', linestyle='--', linewidth=9, alpha = 0.5)
# ax.set_title("Autocorrelation across SNR Levels", fontsize=18)
# only put 0, 20, 40 on the x-axis ticks
# increase the size of the ticks
# Set tick positions
ax.set_xticks([1, 20, 40])
ax.set_yticks([0, 1])
# Set tick label font size
ax.tick_params(axis='both', labelsize=80)
# Disable grid
ax.grid(False)
# plt.tight_layout()
plt.show()

