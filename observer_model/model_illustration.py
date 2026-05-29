# %%
import numpy as np
from scipy import signal as sps
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import acf
from scipy.interpolate import make_interp_spline
from PIL import Image as PILImage


def rms(signal):
    return np.sqrt(np.mean(np.abs(signal) ** 2))


# %%
# ========== SETUP ==========
rng = np.random.default_rng(seed=42)
len_unit = 10
num_reps = 10

# Colors
COLOR_BASE = "#333333A6"       # dark gray for base signal
COLOR_NOISE = '#CC8844'      # amber for noise
COLOR_MIX = "#4B1878"        # purple for mixed signal
COLOR_CRITERION = '#DC143C'  # red for decision criterion

# Generate unit signal and full repeated signal
sig_unit = rng.uniform(-1.0, 1.0, size=len_unit)
signal = np.tile(sig_unit, reps=num_reps)
signal -= signal.mean()
signal /= rms(signal)

# Generate noise (fixed across all coherence levels for consistency)
noise = rng.uniform(-1.0, 1.0, size=signal.shape[0])
noise -= noise.mean()
noise /= rms(noise)


# %%
# =========================
# Plot 1: Repeated Base Signal
# =========================
fig, ax = plt.subplots(figsize=(2, 0.5))

three_reps = signal[:4 * len_unit]

ax.plot(
    three_reps,
    linewidth=3,
    color="gray",
    alpha=0.7
)

ax.set_yticks([])
ax.set_xticks([])

ax.axis("off")
plt.tight_layout(pad =0)

# fig.savefig(
#     'plot_signal.png',
#     dpi=300
# )

plt.show()
# fig.savefig('plot_signal.svg', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
plt.show()
# %%
# =========================
# Plot 2: Noise
# =========================
fig, ax = plt.subplots(figsize=(1, 0.5))

ax.plot(
    noise[:2 * len_unit],
    linewidth=3,
    color=COLOR_NOISE,
    alpha=0.9
)

ax.set_yticks([])
ax.set_xticks([])

ax.axis("off")
plt.tight_layout(pad =0)

fig.savefig('plot_noise.svg', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
plt.show()


# %%
# =========================
# Plot 3: Combined Stimulus
# =========================
coh_example = 0.1  # mid coherence to show a clear mix
stim_example = coh_example * signal + (1 - coh_example) * noise
stim_example -= stim_example.mean()
stim_example /= rms(stim_example)

fig, ax = plt.subplots(figsize=(1.3, 0.2))
ax.plot(
    stim_example[:3* len_unit],
    linewidth=3,
    color='#999999',
    alpha=0.7
)
ax.set_yticks([])
ax.set_xticks([])

ax.axis("off")
plt.tight_layout(pad =0)
fig.savefig('plot_combined_01.svg', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
plt.show()


# %%
coherences = [0.9, 0.5, 0.1]
colors = ['#333333', '#666666', '#999999']
max_lag = 30

acfs = []
for coh in coherences:
    stim = coh * signal + (1 - coh) * noise
    stim -= stim.mean()
    stim /= rms(stim)
    corr, _ = acf(stim, adjusted=False, alpha=0.05, nlags=max_lag)
    acfs.append(corr)

fig, ax = plt.subplots(figsize=(2.5, 3))
n_rows = len(coherences)
row_height = 1

# lag_min, lag_max = 12, 28  # zoom into the peak region

lag_min, lag_max = 6, 14

for i, (coh, corr, col) in enumerate(zip(coherences, acfs, colors)):
    offset = (n_rows - 1 - i) * row_height
    lags = np.arange(1, len(corr))

    # Smooth interpolation
    mask = (lags >= lag_min) & (lags <= lag_max)
    lags_crop = lags[mask]
    corr_crop = corr[1:][mask]
    lags_smooth = np.linspace(lags_crop.min(), lags_crop.max(), 200)
    spline = make_interp_spline(lags_crop, corr_crop, k=3)
    corr_smooth = spline(lags_smooth)

    ax.fill_between(lags_smooth, offset, offset + corr_smooth,
                     color=col, alpha=0.3)
    ax.plot(lags_smooth, offset + corr_smooth, color=col, linewidth=1.5)
    ax.axhline(offset, color='black', linewidth=0.5, zorder=1)

    # Ellipsis dots before and after
    for dot_x in [lag_min - 1.5, lag_min - 2.5, lag_min - 3.5]:
        ax.plot(dot_x, offset + row_height * 0.1, '.', color=col,
                markersize=4)
    for dot_x in [lag_max + 1.5, lag_max + 2.5, lag_max + 3.5]:
        ax.plot(dot_x, offset + row_height * 0.1, '.', color=col,
                markersize=4)

ax.set_yticks([])
ax.set_xlim(lag_min - 5, lag_max + 5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_xticks([])
ax.tick_params(left=False)
plt.rcParams['font.family'] = 'Arial'
plt.tight_layout()
# plt.savefig('acf_schematic.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig('acf_schematic.svg', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
plt.show()



# %%
det_coherences = [0.9, 0.1]
det_labels = ['YES', 'NO']
criterion = 0.45

fig, ax = plt.subplots(figsize=(1.8, 1.3))
n_rows = len(det_coherences)
row_height = 1.1

lag_min, lag_max = 6, 14

for i, (coh, label) in enumerate(zip(det_coherences, det_labels)):
    stim = coh * signal + (1 - coh) * noise
    stim -= stim.mean()
    stim /= rms(stim)
    corr, _ = acf(stim, adjusted=False, alpha=0.05, nlags=max_lag)

    offset = (n_rows - 1 - i) * row_height
    lags = np.arange(1, len(corr))

    mask = (lags >= lag_min) & (lags <= lag_max)
    lags_crop = lags[mask]
    corr_crop = corr[1:][mask]
    lags_smooth = np.linspace(lags_crop.min(), lags_crop.max(), 200)
    spline = make_interp_spline(lags_crop, corr_crop, k=3)
    corr_smooth = spline(lags_smooth)

    col = '#333333' if coh > 0.5 else '#999999'

    ax.fill_between(lags_smooth, offset, offset + corr_smooth,
                    color=col, alpha=0.3)
    ax.plot(lags_smooth, offset + corr_smooth, color=col, linewidth=1.5)
    ax.axhline(offset, color='black', linewidth=0.5, zorder=1)

    # Criterion line
    ax.axhline(offset + criterion, color='#DC143C', linewidth=1.5,
               linestyle='--', alpha=0.8, zorder=2)


    # Ellipsis dots
    for dot_x in [lag_min - 1.5, lag_min - 2.5, lag_min - 3.5]:
        ax.plot(dot_x, offset + row_height * 0.1, '.', color=col,
                markersize=2)
    for dot_x in [lag_max + 1.5, lag_max + 2.5, lag_max + 3.5]:
        ax.plot(dot_x, offset + row_height * 0.1, '.', color=col,
                markersize=2)

ax.set_yticks([])
ax.set_xlim(lag_min - 5, lag_max + 7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_xticks([])
ax.tick_params(left=False)
plt.rcParams['font.family'] = 'Arial'
plt.tight_layout()
# plt.savefig('det_schematic.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig('det_schematic.svg', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
plt.show()

# %%
sms_coherences = [0.9, 0.1]

fig, ax = plt.subplots(figsize=(1.8, 1.3))
n_rows = len(sms_coherences)
row_height = 1.1

# lag_min, lag_max = 12, 28
lag_min, lag_max = 6, 14

for i, coh in enumerate(sms_coherences):
    stim = coh * signal + (1 - coh) * noise
    stim -= stim.mean()
    stim /= rms(stim)
    corr, _ = acf(stim, adjusted=False, alpha=0.05, nlags=max_lag)

    offset = (n_rows - 1 - i) * row_height
    lags = np.arange(1, len(corr))

    mask = (lags >= lag_min) & (lags <= lag_max)
    lags_crop = lags[mask]
    corr_crop = corr[1:][mask]

    # Find peak within visible window
    best_local = lags_crop[np.argmax(corr_crop)]
    max_corr = corr[best_local]

    # Smooth interpolation
    lags_smooth = np.linspace(lags_crop.min(), lags_crop.max(), 200)
    spline = make_interp_spline(lags_crop, corr_crop, k=3)
    corr_smooth = spline(lags_smooth)

    col = '#333333' if coh > 0.5 else '#999999'

    ax.fill_between(lags_smooth, offset, offset + corr_smooth,
                    color=col, alpha=0.3)
    ax.plot(lags_smooth, offset + corr_smooth, color=col, linewidth=1.5)
    ax.axhline(offset, color='black', linewidth=0.5, zorder=1)

    # Dot at the peak
    ax.plot(best_local, offset + max_corr, 'o', color="#0A30AF", markersize=5,
            markeredgecolor='black', markeredgewidth=0.5, zorder=3)


    # Ellipsis dots
    for dot_x in [lag_min - 1.5, lag_min - 2.5, lag_min - 3.5]:
        ax.plot(dot_x, offset + row_height * 0.1, '.', color=col,
                markersize=2)
    for dot_x in [lag_max + 1.5, lag_max + 2.5, lag_max + 3.5]:
        ax.plot(dot_x, offset + row_height * 0.1, '.', color=col,
                markersize=2)

ax.set_yticks([])
ax.set_xlim(lag_min - 5, lag_max + 7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['left'].set_visible(False)
ax.set_xticks([])
ax.tick_params(left=False)
plt.rcParams['font.family'] = 'Arial'
plt.tight_layout()
# plt.savefig('sms_schematic.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig('sms_schematic.svg', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
plt.show()



# %%
det_coherences = [0.9, 0.1]
det_labels = ['YES', 'NO']
criterion = 0.45

fig, ax = plt.subplots(figsize=(4, 1.8))
n_rows = len(det_coherences)
row_height = 1.3

for i, (coh, label) in enumerate(zip(det_coherences, det_labels)):
    stim = coh * signal + (1 - coh) * noise
    stim -= stim.mean()
    stim /= rms(stim)
    corr, _ = acf(stim, adjusted=False, alpha=0.05, nlags=max_lag)

    offset = (n_rows - 1 - i) * row_height
    lags = np.arange(1, len(corr))
    col = '#333333' if coh > 0.5 else '#999999'
    ax.bar(lags, corr[1:], bottom=offset, color=col, alpha=0.85, width=0.9,
           edgecolor='none')
    ax.axhline(offset, color='black', linewidth=1, zorder=1)
    ax.axhline(offset + criterion, color='#DC143C', linewidth=2,
               linestyle='--', alpha=0.8, zorder=2)
#     ax.text(41, offset + criterion + 0.05, 'criterion', fontsize=20,
#             fontstyle='italic', color='#DC143C', va='bottom')

    # Right margin: coherence value
    # ax.text(87, offset + row_height * 0.3, f'{coh}', fontsize=20,
    #         fontweight='bold', va='center', ha='left', color=col,
    #         clip_on=False)

# "Coherence" header above the right margin values
# ax.text(87, (n_rows - 1) * row_height + row_height * 0.3 + 0.6,
#         'Coherence', fontsize=20, fontweight='bold', va='bottom', ha='center',
#         color='black', clip_on=False, fontstyle='italic')

# YES/NO further to the right, past the coherence values
# for i, label in enumerate(det_labels):
#     offset = (n_rows - 1 - i) * row_height
#     ax.text(48, offset + row_height * 0.3, label, fontsize=20, fontweight='bold',
#             va='center', ha='left',
#             color='black' if label == 'YES' else "black",
#             clip_on=False)


ax.set_yticks([])
ax.set_xlim(0, max_lag)
# ax.set_xlabel('Lag', fontsize=30, fontweight='bold')
# ax.set_ylabel('ACF', fontsize=30, fontweight='bold', rotation=90, labelpad=13)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_xticks([])
ax.tick_params(left=False)
# ax.set_title("Detection Task", fontsize=30, fontweight='bold', y=1)
plt.tight_layout()
# plt.subplots_adjust(right=0.85)
plt.show()
fig.savefig('plot_acf_detection.png', dpi=300)  # no bbox_inches='tight'


# %%
sms_coherences = [0.9, 0.1]

fig, ax = plt.subplots(figsize=(4, 1.8))
n_rows = len(sms_coherences)
row_height = 1.3

for i, coh in enumerate(sms_coherences):
    stim = coh * signal + (1 - coh) * noise
    stim -= stim.mean()
    stim /= rms(stim)
    corr, _ = acf(stim, adjusted=False, alpha=0.05, nlags=max_lag)

    peak_idx, _ = sps.find_peaks(corr[1:], height=0)
    if peak_idx.size == 0:
        best_idx = 1
    else:
        best_idx = peak_idx[np.argmax(corr[1:][peak_idx])] + 1
    max_corr = corr[best_idx]

    offset = (n_rows - 1 - i) * row_height
    lags = np.arange(1, len(corr))
    col = '#333333' if coh > 0.5 else '#999999'
    ax.bar(lags, corr[1:], bottom=offset, color=col, alpha=0.85, width=0.8,
           edgecolor='none')
    ax.axhline(offset, color='black', linewidth=1, zorder=1)

    # Dot at the maximum peak
    ax.plot(best_idx, offset + max_corr, 'o', color="#0A30AF", markersize=5,
            markeredgecolor='black', markeredgewidth=1, zorder=3)
    # Value written next to the dot
    # ax.text(best_idx + 2, offset + max_corr, f'{max_corr:.2f}', fontsize=13,
    #         fontweight='bold', color='#069AF3', va='center', ha='left')

    # Right margin: coherence value
#     ax.text(46, offset + row_height * 0.3, f'{max_corr:.2f}', fontsize=20,
#             fontweight='bold', va='center', ha='left', color="#0A30AF",
#             clip_on=False)

# "Coherence" header above the right margin values
# ax.text(46, (n_rows - 1) * row_height + row_height * 0.3 + 0.4,
#         'Max Corr', fontsize=20, fontweight='bold', va='bottom', ha='center',
#         color='black', clip_on=False, fontstyle='italic')

ax.set_yticks([])
ax.set_xlim(0, max_lag)
# ax.set_xlabel('Lag', fontsize=30, fontweight='bold')
# ax.set_ylabel('ACF', fontsize=30, fontweight='bold', rotation=90, labelpad=13)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_xticks([])
ax.tick_params(left=False)
# ax.set_title("SMS Task", fontsize=30, fontweight='bold', y=1)
plt.tight_layout()
# plt.subplots_adjust(right=0.85)
plt.show()
fig.savefig('plot_acf_sms.png', dpi=300)  # no bbox_inches='tight'


