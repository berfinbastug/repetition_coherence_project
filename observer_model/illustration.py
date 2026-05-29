# %%
import numpy as np
from scipy import signal as sps
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import acf
from scipy.interpolate import make_interp_spline
from PIL import Image as PILImage

# Read the reference figure's actual height
ref_img = PILImage.open('plot2.png')  # or plot1.png, they're the same
REF_HEIGHT_IN = ref_img.size[1] / 300  # divide pixels by your DPI
print(f"Reference height: {REF_HEIGHT_IN:.2f} inches")


def rms(signal):
    return np.sqrt(np.mean(np.abs(signal) ** 2))


# %%
# ========== SETUP ==========
rng = np.random.default_rng(seed=42)
len_unit = 20
num_reps = 6

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
fig, ax = plt.subplots(figsize=(1, 0.5))

three_reps = signal[:2 * len_unit]

ax.plot(
    three_reps,
    linewidth=3,
    color="gray",
    alpha=0.7
)

# for x in [len_unit, 2 * len_unit]:
#     ax.axvline(
#         x=x,
#         color='black',
#         linewidth=4,
#         linestyle='--',
#         alpha=0.8
#     )

# ax.set_title(
#     "Repeated Base Signal",
#     fontsize=30,
#     fontweight='bold',
#     y=1.05
# )

ax.set_yticks([])
ax.set_xticks([])

# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

ax.axis("off")
plt.tight_layout(pad =0)

fig.savefig(
    'plot_signal.png',
    dpi=300
)

plt.show()
fig.savefig('plot_signal.svg', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
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

# ax.set_title(
#     "Noise",
#     fontsize=30,
#     fontweight='bold',
#     y=1.05
# )

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
coh_example = 0.8  # mid coherence to show a clear mix
stim_example = coh_example * signal + (1 - coh_example) * noise
stim_example -= stim_example.mean()
stim_example /= rms(stim_example)

fig, ax = plt.subplots(figsize=(1, 0.5))
ax.plot(
    stim_example[:2 * len_unit],
    linewidth=3,
    color=COLOR_MIX,
    alpha=0.7
)
ax.set_yticks([])
ax.set_xticks([])

ax.axis("off")
plt.tight_layout(pad =0)
fig.savefig('plot_combined.svg', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
plt.show()


# %%
coherences = [0.9, 0.5, 0.1]
colors = ['#333333', '#666666', '#999999']
max_lag = 45

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

lag_min, lag_max = 12, 28  # zoom into the peak region

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

fig, ax = plt.subplots(figsize=(2.2, 1.4))
n_rows = len(det_coherences)
row_height = 1

lag_min, lag_max = 12, 28

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

#     # YES/NO label to the right
#     ax.text(lag_max + 5, offset + row_height * 0.4, label, fontsize=7,
#             fontweight='bold', va='center', ha='left',
#             color='#006400' if label == 'YES' else '#DC143C',
#             clip_on=False)

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

fig, ax = plt.subplots(figsize=(2.2, 1.4))
n_rows = len(sms_coherences)
row_height = 1

lag_min, lag_max = 12, 28

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

#     # Max corr value to the right
#     ax.text(lag_max + 5, offset + row_height * 0.4, f'{max_corr:.2f}',
#             fontsize=7, fontweight='bold', va='center', ha='left',
#             color='#0A30AF', clip_on=False)

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




# %%
fig, ax = plt.subplots(figsize=(9, 1))
ax.axis('off')

# Build the equation with colored words
ax.text(0.2, 0.5,
        'coherence ', fontsize= 25, fontweight='bold', color='black',
        ha='right', va='center', transform=ax.transAxes)
ax.text(0.21, 0.5,
        '×', fontsize=25, color='black',
        ha='center', va='center', transform=ax.transAxes)
ax.text(0.23, 0.5,
        ' signal', fontsize=25, fontweight='bold', color='gray',
        ha='left', va='center', transform=ax.transAxes)
ax.text(0.41, 0.5,
        '+ ', fontsize=25, color='black',
        ha='left', va='center', transform=ax.transAxes)
ax.text(0.46, 0.5, '(1 − ', fontsize=25, color='black',
        ha='left', va='center', transform=ax.transAxes)
ax.text(0.55, 0.5, ' coherence', fontsize=25, color='black', fontweight='bold',
        ha='left', va='center', transform=ax.transAxes)
ax.text(0.83, 0.5, ' ) ×', fontsize=25, color='black',
        ha='left', va='center', transform=ax.transAxes)
ax.text(0.9, 0.5,
        ' noise', fontsize=25, fontweight='bold', color='#CC8844',
        ha='left', va='center', transform=ax.transAxes)

plt.tight_layout()
plt.savefig('mixing_equation.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()



# %%
fig, ax = plt.subplots(figsize=(1, 10))

ax.text(
    0.5, 0.5,
    'Decreasing coherence',
    fontsize=30,
    fontweight='bold',
    rotation=270,
    va='center',
    ha='center'
)

ax.axis('off')

plt.tight_layout()
plt.savefig('acf_label.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# %%
fig, axes = plt.subplots(2, 1, figsize=(3, 4), sharex=True)

# =========================
# Top: Repeated Base Signal
# =========================
ax = axes[0]

three_reps = signal[:3 * len_unit]

ax.plot(
    three_reps,
    linewidth=5,
    color="gray",
    alpha=0.7
)

for x in [len_unit, 2 * len_unit]:
    ax.axvline(
        x=x,
        color='black',
        linewidth=4,
        linestyle='--',
        alpha=0.8
    )

# ax.set_title(
#     "Repeated Base Signal",
#     fontsize=30,
#     fontweight='bold',
#     y=1.05
# )

ax.set_yticks([])
ax.set_xticks([])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# =========================
# Bottom: Noise
# =========================
ax = axes[1]

ax.plot(
    noise[:3 * len_unit],
    linewidth=5,
    color=COLOR_NOISE,
    alpha=0.9
)

# ax.set_title(
#     "Noise",
#     fontsize=30,
#     fontweight='bold',
#     y=1.05
# )

ax.set_yticks([])
ax.set_xticks([])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# =========================
# Layout
# =========================
plt.tight_layout()

fig.savefig(
    'plot_signal_noise.png',
    dpi=300
)

plt.show()


# %%
coherences = [0.9, 0.5, 0.1]
colors = ['#333333', '#666666', '#999999']

max_lag = 45

acfs = []
for coh in coherences:
    stim = coh * signal + (1 - coh) * noise
    stim -= stim.mean()
    stim /= rms(stim)
    corr, _ = acf(stim, adjusted=False, alpha=0.05, nlags=max_lag)
    acfs.append(corr)

fig, ax = plt.subplots(figsize=(5, 4))
n_rows = len(coherences)
row_height = 1.3

for i, (coh, corr, col) in enumerate(zip(coherences, acfs, colors)):
    offset = (n_rows - 1 - i) * row_height
    lags = np.arange(1, len(corr))
    ax.bar(lags, corr[1:], bottom=offset, color=col, alpha=0.85, width=0.9,
           edgecolor='none')
    ax.axhline(offset, color='black', linewidth=1, zorder=1)

#     # Right margin: coherence value
#     ax.text(87, offset + row_height * 0.15, f'{coh}', fontsize=25,
#             fontweight='bold', va='center', ha='left', color=col,
#             clip_on=False)

# # "Coherence" label above the right margin values
# ax.text(87, (n_rows - 1) * row_height + row_height * 0.2 + 0.6,
#         'Coherence', fontsize=25, fontweight='bold', va='bottom', ha='center',
#         color='black', clip_on=False, fontstyle='italic')

ax.set_yticks([])
ax.set_xlim(0, 45)
# ax.set_xlabel('Lag', fontsize=30, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_xticks([])
ax.tick_params(left=False)
# ax.set_ylabel('ACF', fontsize=30, fontweight='bold', rotation=90, labelpad=20)
plt.tight_layout()
plt.subplots_adjust(right=0.98)
plt.savefig('acf.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# %%
# # ========== PANEL 5: DETECTION READOUT ==========
# # Show two ACF plots side by side: one where peak > criterion (YES), one where peak < criterion (NO)
# # Use coherence 0.7 (detected) and 0.1 (not detected)
# coherences = [0.9, 0.5, 0.1]
 
# acf_data = {}
# for coh in coherences:
#     stim = coh * signal + (1 - coh) * noise
#     stim -= stim.mean()
#     stim /= rms(stim)
#     corr, confint = acf(stim, adjusted=False, alpha=0.05, nlags=stim.shape[0] - 1)
#     peak_idx, _ = sps.find_peaks(corr, height=0)
#     if peak_idx.size == 0:
#         peak_idx = np.array([np.random.randint(1, corr.size)])
#     best_idx = peak_idx[np.argmax(corr[peak_idx])]
#     acf_data[coh] = {
#         'corr': corr,
#         'best_idx': best_idx,
#         'max_corr': corr[best_idx],
#     }
# criterion = 0.45

# fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)

# # --- YES case (high coherence) ---
# ax = axes[0]
# d = acf_data[0.9]
# lags = np.arange(len(d['corr']))
# ax.bar(lags, d['corr'], color=COLOR_MIX, alpha=0.7, width=0.8)
# ax.axhline(criterion, color=COLOR_CRITERION, linewidth=2.5, linestyle='--',
#            label=f'Criterion = {criterion}')
# ax.axvline(x=len_unit, color='#2ECC71', linewidth=1.5, linestyle=':', alpha=0.5)

# # Arrow pointing to the peak that exceeds criterion
# ax.annotate('peak > criterion',
#             xy=(d['best_idx'], d['max_corr']),
#             xytext=(d['best_idx'] + 12, d['max_corr'] + 0.05),
#             fontsize=11, fontweight='bold', color='#006400',
#             arrowprops=dict(arrowstyle='->', color='#006400', lw=1.5))

# ax.set_title('Detection: YES', fontsize=16, fontweight='bold', color='#006400')
# ax.set_ylabel('Auto Corr', fontsize=12)
# ax.set_xlabel('Lag', fontsize=12)
# ax.set_xlim(0, 80)
# ax.set_ylim(-0.3, 1.05)
# ax.legend(fontsize=10, loc='upper right')
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.tick_params(labelsize=10)

# # --- NO case (low coherence) ---
# ax = axes[1]
# d = acf_data[0.1]
# lags = np.arange(len(d['corr']))
# ax.bar(lags, d['corr'], color=COLOR_MIX, alpha=0.4, width=0.8)
# ax.axhline(criterion, color=COLOR_CRITERION, linewidth=2.5, linestyle='--',
#            label=f'Criterion = {criterion}')
# ax.axvline(x=len_unit, color='#2ECC71', linewidth=1.5, linestyle=':', alpha=0.5)

# # Arrow pointing to the peak that doesn't exceed criterion
# ax.annotate('peak < criterion',
#             xy=(d['best_idx'], d['max_corr']),
#             xytext=(d['best_idx'] + 12, d['max_corr'] + 0.15),
#             fontsize=11, fontweight='bold', color=COLOR_CRITERION,
#             arrowprops=dict(arrowstyle='->', color=COLOR_CRITERION, lw=1.5))

# ax.set_title('Detection: NO', fontsize=16, fontweight='bold', color=COLOR_CRITERION)
# ax.set_xlabel('Lag', fontsize=12)
# ax.set_xlim(0, 80)
# ax.legend(fontsize=10, loc='upper right')
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.tick_params(labelsize=10)

# plt.tight_layout()

# plt.show()


# %%
# # ========== PANEL 6: SMS READOUT ==========
# # Show two ACF plots side by side: high coherence (strong peak = good tracking)
# # and low coherence (weak peak = poor tracking). Highlight the peak magnitude.

# fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)

# # --- High coherence SMS ---
# ax = axes[0]
# d = acf_data[0.7]
# lags = np.arange(len(d['corr']))
# ax.bar(lags, d['corr'], color=COLOR_MIX, alpha=0.7, width=0.8)
# ax.axvline(x=len_unit, color='#2ECC71', linewidth=1.5, linestyle=':', alpha=0.5)

# # Highlight the peak with a horizontal line at peak height
# ax.axhline(d['max_corr'], color='#069AF3', linewidth=2, linestyle='-', alpha=0.6)
# ax.annotate(f'peak magnitude = {d["max_corr"]:.2f}\n→ strong synchronization',
#             xy=(d['best_idx'], d['max_corr']),
#             xytext=(d['best_idx'] + 10, d['max_corr'] - 0.2),
#             fontsize=11, fontweight='bold', color='#069AF3',
#             arrowprops=dict(arrowstyle='->', color='#069AF3', lw=1.5))

# ax.set_title('SMS: High Coherence', fontsize=16, fontweight='bold', color='#069AF3')
# ax.set_ylabel('Auto Corr', fontsize=12)
# ax.set_xlabel('Lag', fontsize=12)
# ax.set_xlim(0, 80)
# ax.set_ylim(-0.3, 1.05)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.tick_params(labelsize=10)

# # --- Low coherence SMS ---
# ax = axes[1]
# d = acf_data[0.1]
# lags = np.arange(len(d['corr']))
# ax.bar(lags, d['corr'], color=COLOR_MIX, alpha=0.4, width=0.8)
# ax.axvline(x=len_unit, color='#2ECC71', linewidth=1.5, linestyle=':', alpha=0.5)

# ax.axhline(d['max_corr'], color='#069AF3', linewidth=2, linestyle='-', alpha=0.6)
# ax.annotate(f'peak magnitude = {d["max_corr"]:.2f}\n→ weak synchronization',
#             xy=(d['best_idx'], d['max_corr']),
#             xytext=(d['best_idx'] + 10, d['max_corr'] + 0.2),
#             fontsize=11, fontweight='bold', color='#069AF3',
#             arrowprops=dict(arrowstyle='->', color='#069AF3', lw=1.5))

# ax.set_title('SMS: Low Coherence', fontsize=16, fontweight='bold', color='#069AF3')
# ax.set_xlabel('Lag', fontsize=12)
# ax.set_xlim(0, 80)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.tick_params(labelsize=10)

# plt.tight_layout()
# plt.savefig('/mnt/user-data/outputs/fig_06_sms_readout.png', dpi=200, bbox_inches='tight',
#             facecolor='white')
# plt.savefig('/mnt/user-data/outputs/fig_06_sms_readout.svg', bbox_inches='tight',
#             facecolor='white')
# plt.show()

# print("All panels saved as PNG and SVG!")





# %%
# COLOR_CRITERION = '#DC143C'
# COLOR_PEAK_LINE = '#069AF3'

# coherences = [1.0, 0.7, 0.4, 0.1]
 
# acf_data = {}
# for coh in coherences:
#     stim = coh * signal + (1 - coh) * noise
#     stim -= stim.mean()
#     stim /= rms(stim)
#     corr, confint = acf(stim, adjusted=False, alpha=0.05, nlags=stim.shape[0] - 1)
#     peak_idx, _ = sps.find_peaks(corr, height=0)
#     if peak_idx.size == 0:
#         peak_idx = np.array([np.random.randint(1, corr.size)])
#     best_idx = peak_idx[np.argmax(corr[peak_idx])]
#     acf_data[coh] = {
#         'corr': corr,
#         'best_idx': best_idx,
#         'max_corr': corr[best_idx],
#     }
 


# # %%
# # ========== PLOT ==========
# fig, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=True,
#                           gridspec_kw={'hspace': 0.4})
 
# for i, (coh, ax) in enumerate(zip(coherences, axes)):
#     d = acf_data[coh]
#     lags = np.arange(len(d['corr']))
 
#     # Bar plot (skip lag 0)
#     ax.bar(lags[1:], d['corr'][1:], color='black', alpha=0.9, width=0.8)
 
#     # Zero line
#     ax.axhline(0, color='gray', linewidth=0.5)
 
#     # Horizontal line at peak value, extending from the peak bar to the right edge
#     ax.axhline(d['max_corr'], color=COLOR_PEAK_LINE, linewidth=2,
#                linestyle='-', alpha=0.6)
 
#     # Peak value label at the right end of the horizontal line
#     ax.text(78, d['max_corr'] + 0.04, f'{d["max_corr"]:.2f}',
#             fontsize=12, fontweight='bold', color=COLOR_PEAK_LINE,
#             ha='right', va='bottom')
 
#     # Coherence label
#     ax.set_title(f'Coherence = {coh}', fontsize=14, fontweight='bold', loc='center')
 
#     # Axis limits
#     ax.set_ylim(-0.3, 1.05)
#     ax.set_xlim(0, 80)
 
#     # Clean spines
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.tick_params(labelsize=11)
 
#     # Remove individual y-axis labels (shared label below)
#     ax.set_ylabel('')
 
# # Shared y-axis label
# fig.text(0.02, 0.5, 'Autocorrelation', fontsize=16, fontweight='bold',
#          va='center', ha='center', rotation=90)
 
# # Shared x-axis label (only on the bottom subplot)
# axes[-1].set_xlabel('Lag', fontsize=16, fontweight='bold')
 
# # plt.savefig('fig_04_acf_coherences.png', dpi=200, bbox_inches='tight',
# #             facecolor='white')
# # plt.savefig('fig_04_acf_coherences.svg', bbox_inches='tight',
# #             facecolor='white')
# plt.show()




# # ========== PANEL 2: REPEATED BASE SIGNAL (3 units shown, vertical separators) ==========
# fig, ax = plt.subplots(figsize=(10, 4))
# three_reps = signal[:3 * len_unit]
# ax.plot(three_reps, linewidth=5, color=COLOR_BASE, alpha=0.9)

# # Vertical lines at repetition boundaries
# for x in [len_unit, 2 * len_unit]:
#     ax.axvline(x=x, color='gray', linewidth=4, linestyle='--', alpha=0.8)

# # Label each repetition
# # for i in range(3):
# #     mid = i * len_unit + len_unit / 2
# #     ax.text(mid, ax.get_ylim()[1] * 0.9, f'cycle {i+1}', ha='center', fontsize=14,
# #             color='#666', fontstyle='italic')

# ax.set_title("Repeated Base Signal", fontsize=30,
#              fontweight='bold', y=1.3)
# ax.set_yticks([])
# ax.set_ylabel("Amplitude", fontsize= 30)


# ax.set_xticks([0, 20, 40, 60])  # positions only
# ax.tick_params(axis='x', labelsize=30)  # control label size
# ax.set_xlabel("Samples", fontsize= 30)

# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.tight_layout()
# plt.show()


# # %%
# # ========== PANEL 3: NOISE SIGNAL (same length as 3 units) ==========
# fig, ax = plt.subplots(figsize=(10, 4))
# ax.plot(noise[:3 * len_unit], linewidth=5, color=COLOR_NOISE, alpha=0.9)
# ax.set_title("Noise", fontsize=20, fontweight='bold', y=1.3)
# ax.set_ylabel("Amplitude", fontsize=30)
# ax.set_xticks([0, 20, 40, 60])  # positions only
# ax.tick_params(axis='x', labelsize=30)  # control label size
# ax.set_xlabel("Samples", fontsize= 30)

# ax.set_yticks([])
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.tight_layout()
# plt.show()


# %%
# ========== PANEL 1: BASE SIGNAL (1 unit) ==========
# fig, ax = plt.subplots(figsize=(4, 3))
# ax.plot(signal[:len_unit], linewidth=4, color=COLOR_BASE, alpha=0.9)
# ax.set_title("Base Signal (1 unit)", fontsize=20, fontweight='bold', y=1.02)
# ax.set_ylabel("Amplitude", fontsize=12)
# ax.set_xticks([])
# ax.set_yticks([])
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.tight_layout()
# plt.show()



# %%
# coherences = [0.9, 0.5, 0.1]
# colors = ['#333333', '#666666', '#999999']

# acfs = []

# for coh in coherences:
#     stim = coh * signal + (1 - coh) * noise
#     stim -= stim.mean()
#     stim /= rms(stim)

#     corr, _ = acf(
#         stim,
#         adjusted=False,
#         alpha=0.05,
#         nlags=stim.shape[0] - 1
#     )

#     acfs.append(corr)

# # =========================
# # Plot
# # =========================
# fig, ax = plt.subplots(figsize=(12, 9))

# n_rows = len(coherences)
# row_height = 1.7

# for i, (coh, corr, col) in enumerate(zip(coherences, acfs, colors)):

#     # Vertical offset for stacking
#     offset = (n_rows - 1 - i) * row_height

#     lags = np.arange(1, len(corr))

#     # ACF bars
#     ax.bar(
#         lags,
#         corr[1:],
#         bottom=offset,
#         color=col,
#         alpha=0.85,
#         width=0.8,
#         edgecolor='none'
#     )

#     # Baseline for each row
#     ax.axhline(
#         offset,
#         color='black',
#         linewidth=1,
#         zorder=1
#     )

# # =========================
# # Arrow showing decreasing coherence
# # =========================
# x_arrow = 95

# y_top = (n_rows - 1) * row_height + 0.9
# y_bottom = -0.05

# ax.annotate(
#     '',
#     xy=(x_arrow, y_bottom),      # arrow head
#     xytext=(x_arrow, y_top),     # arrow start
#     arrowprops=dict(
#         arrowstyle='->',
#         linewidth=5,
#         color='black'
#     ),
#     annotation_clip=False
# )

# # Vertical label
# ax.text(
#     x_arrow + 4,
#     (y_top + y_bottom) / 2,
#     'Decreasing\ncoherence',
#     fontsize=24,
#     fontweight='bold',
#     rotation=270,
#     va='center',
#     ha='center'
# )

# # =========================
# # Axis styling
# # =========================
# ax.set_xlim(0, 90)

# ax.set_yticks([])

# ax.set_xlabel(
#     'Lag',
#     fontsize=30,
#     fontweight='bold'
# )

# ax.set_ylabel(
#     'ACF',
#     fontsize=30,
#     fontweight='bold',
#     rotation=90,
#     labelpad=15
# )

# ax.set_xticks([])

# ax.tick_params(left=False)

# # Remove unnecessary spines
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['left'].set_visible(False)

# # Layout
# plt.tight_layout()
# plt.subplots_adjust(right=0.92)

# # Save
# plt.savefig(
#     'acf.png',
#     dpi=300,
#     bbox_inches='tight',
#     facecolor='white'
# )

# plt.show()



# %%
# fig, axes = plt.subplots(1, 2, figsize=(12, 3), sharey=True)

# # Left: Repeated Base Signal
# ax = axes[0]
# three_reps = signal[:3 * len_unit]
# ax.plot(three_reps, linewidth=5, color= "gray", alpha=0.7)
# for x in [len_unit, 2 * len_unit]:
#     ax.axvline(x=x, color='black', linewidth=4, linestyle='--', alpha=0.8)
# ax.set_title("Repeated Base Signal", fontsize=30, fontweight='bold', y=1.1)
# # ax.set_ylabel("Amplitude", fontsize=30)
# ax.set_yticks([])
# ax.set_xticks([])
# ax.tick_params(axis='x', labelsize=30)
# # ax.set_xlabel("Samples", fontsize=30)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

# # Right: Noise
# ax = axes[1]
# ax.plot(noise[:3 * len_unit], linewidth=5, color=COLOR_NOISE, alpha=0.9)
# ax.set_title("Noise", fontsize=30, fontweight='bold', y=1.1)
# ax.set_yticks([])
# ax.set_xticks([])
# ax.tick_params(axis='x', labelsize=30)
# # ax.set_xlabel("Samples", fontsize=30)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

# plt.tight_layout()
# fig.savefig('plot_signal_noise.png', dpi=300)  # no bbox_inches='tight'
# plt.show()