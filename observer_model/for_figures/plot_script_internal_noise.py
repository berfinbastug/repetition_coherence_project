# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase
from scipy.stats import sem
from PIL import Image as PILImage



# %%
# ========== SETTINGS ==========
RANDOM_SEED = 42
period_min, period_max = 20, 40
period_domain = np.logspace(np.log10(period_min), np.log10(period_max),
                            num=3, dtype=int)

period_colors = {period_domain[0]: '#069AF3',
                 period_domain[1]: '#006400',
                 period_domain[2]: '#DC143C'}
period_labels = {period_domain[0]: 'short',
                 period_domain[1]: 'mid',
                 period_domain[2]: 'long'}

base_colors = {period_domain[0]: "#069AF3",
               period_domain[1]: "#006400",
               period_domain[2]: "#DC143C"}

DPI = 300

def get_colormap(base_hex, n_levels):
    rgb = np.array(mcolors.to_rgb(base_hex))
    white = np.array([1, 1, 1])
    return [mcolors.to_hex((1 - t) * white + t * rgb)
            for t in np.linspace(0.0, 1.0, n_levels)]

# %%
# ========== READ DATA ==========
df_detect = pd.read_csv(f'simulation_results/ncycle_sim_results_internal_noise_scaled_seed{RANDOM_SEED}.csv')
df_wide = pd.read_csv(f'simulation_results/acf_model_simulations_internal_noise_scaled_seed{RANDOM_SEED}.csv')
df_wide["Decision"] = df_wide["Decision"].astype(int)
df_wide["Autocorr"] = pd.to_numeric(df_wide["Autocorr"], errors='coerce')


# %%
# ==========================================================================
# STEP 1: Build and save FIGURE 2 first (the reference with colorbars)
#         This is your ORIGINAL figure 2 code, untouched.
# ==========================================================================
unique_periods = sorted(df_wide["Period"].unique())
selected_periods = [unique_periods[0], unique_periods[len(unique_periods)//2],
                    unique_periods[-1]]

all_snrs = sorted(df_wide["SNR"].unique())
num_snr_levels = 10
selected_indices = np.round(np.linspace(0, len(all_snrs) - 1, num_snr_levels)).astype(int)
snrs = [all_snrs[i] for i in selected_indices]
snr_to_index = {snr: i for i, snr in enumerate(snrs)}

LINEWIDTH = 1.5
fig2, axs = plt.subplots(1, 3, figsize=(5.2, 1.3), sharey=True)

for i, period in enumerate(selected_periods):
    ax = axs[i]
    sub_df = df_wide[df_wide["Period"] == period]
    colormap = get_colormap(base_colors[period], len(snrs))

    for snr in snrs:
        df_plot = (sub_df[sub_df["SNR"] == snr]
                   .groupby("Reps")["Autocorr"].mean().reset_index())
        color = colormap[snr_to_index[snr]]
        ax.plot(df_plot["Reps"], df_plot["Autocorr"],
                color=color, linewidth = LINEWIDTH)
        ax.scatter(df_plot["Reps"], df_plot["Autocorr"],
                   color=color, s=5, edgecolor='black', linewidth= LINEWIDTH, alpha=0.6)

    ax.set_xticks([])
    ax.set_yticks([])
    # ax.tick_params(axis='both', labelsize=30)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    # ax.set_xlabel('N Cycles', fontsize=30, fontweight='bold')
    # ax.set_title(period_labels[period], fontsize=30, fontweight='bold',
    #              color=base_colors[period])
    # if i == 0:
    #     ax.set_ylabel('Max Corr', fontsize=30, fontweight='bold')

plt.tight_layout()
plt.subplots_adjust(right=0.75, wspace=0.3)

for i, period in enumerate(selected_periods):
    ax = axs[i]
    snr_norm = mcolors.Normalize(vmin=min(snrs), vmax=max(snrs))
    box = ax.get_position()
    cbar_ax = fig2.add_axes([box.x1 + 0.01, box.y0, 0.015, box.height])
    cmap_local = mcolors.LinearSegmentedColormap.from_list(
        "custom", get_colormap(base_colors[period], 256))
    cb = ColorbarBase(cbar_ax, cmap=cmap_local, norm=snr_norm, orientation='vertical')

    if i == len(selected_periods) - 1:
        # cb.ax.tick_params(labelsize=20)
        cb.ax.set_yticks([])
        # cb.set_label('Coherence', fontsize=20, fontweight='bold')
    else:
        cb.ax.set_yticks([])


fig2.savefig('plot2_sensory_noise.svg', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
plt.show()




# %%
# ==========================================================================
# STEP 2: Build FIGURE 1 with figsize derived from figure 2's actual pixels
#         Save WITHOUT bbox_inches='tight' so figsize controls output exactly
# ==========================================================================
fig1, axes = plt.subplots(1, 3, figsize=(4.2, 1.3))

LINEWIDTH = 1.3
MARKER_SIZE_DECISION = 6
MARKER_SIZE = 2

# --- Panel A: Decision Criterion ---
ax = axes[0]
base_noise = 0.15  # your base internal noise level
noise_slope = 0.6

periods_dense = np.linspace(period_min, period_max, 100)
norm_dense = np.log(periods_dense / period_min) / np.log(period_max / period_min)
noise_dense = base_noise + norm_dense * noise_slope

ax.plot(periods_dense, noise_dense, color='gray', linewidth=LINEWIDTH, alpha=0.7)
for period in period_domain:
    norm_scale = np.log(period / period_min) / np.log(period_max / period_min)
    noise_val = base_noise + norm_scale * noise_slope
    ax.plot(period, noise_val, 'o', color=period_colors[period], markersize=MARKER_SIZE_DECISION,
            markeredgecolor='black', markeredgewidth=1, zorder=3)

ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- Panel B: Psychometric Curve ---
ax = axes[1]
df_avg = df_wide.groupby(["SNR", "Period"])["Decision"].mean().reset_index()

for period in period_domain:
    sub_df = df_avg[df_avg['Period'] == period]
    ax.plot(sub_df['SNR'], sub_df['Decision'],
            color=period_colors[period], linewidth = LINEWIDTH, marker='o', markersize = MARKER_SIZE)

# ax.set_xlabel('Coherence', fontsize=30, fontweight='bold')
# ax.set_ylabel('P(Yes)', fontsize=30, fontweight='bold')
ax.set_yticks([])
ax.set_xticks([])
# ax.set_yticks([0, 0.5, 1])
# ax.tick_params(axis='both', labelsize=14)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(False)

# --- Panel C: Cycles to Detection ---
ax = axes[2]
summary = df_detect.groupby(['SNR', 'Period']).agg(
    mean_ncycle=('N_Cycles', 'mean'),
    sem_ncycle=('N_Cycles', sem),
    mean_detected=('Detected', 'mean')
).reset_index()

summary['mean_detected'] = summary['mean_detected'].fillna(0)
min_detect = summary['mean_detected'].min()
max_detect = summary['mean_detected'].max()
range_detect = max_detect - min_detect if max_detect > min_detect else 1
summary['correct_norm'] = ((summary['mean_detected'] - min_detect)
                           / range_detect).fillna(0)

for period in period_domain:
    sub = summary[summary['Period'] == period].sort_values('SNR')
    if sub.empty:
        continue
    x = sub['SNR']
    y = sub['mean_ncycle']
    color = period_colors[period]
    for i in range(len(sub) - 1):
        x_pair = x.iloc[i:i + 2]
        y_pair = y.iloc[i:i + 2]
        alpha = float(np.clip(sub['correct_norm'].iloc[i], 0, 1))
        ax.plot(x_pair, y_pair, color=color, alpha=alpha, linewidth = MARKER_SIZE)

# ax.set_xlabel('Coherence', fontsize=30, fontweight='bold')
# ax.set_ylabel('N Cycles', fontsize=30, fontweight='bold')
ax.set_yticks([])
ax.set_xticks([])
# ax.tick_params(axis='both', labelsize=14)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(False)

# Use constrained_layout=False, manually adjust to fill the figure nicely
plt.tight_layout()
# plt.subplots_adjust(wspace=0.3)
# Save WITHOUT bbox_inches='tight' — figsize already matches target pixels
# fig1.savefig('plot1.png', dpi=DPI)
fig1.savefig('plot1_sensory_noise.svg', transparent=True, bbox_inches='tight', pad_inches=0)
plt.show()



# %%
