# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase
from scipy.stats import sem
from matplotlib.transforms import Bbox

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

fixed_bbox = Bbox([[0, 0], [21, 5]])  # in inches, adjust to your figsize


def get_colormap(base_hex, n_levels):
    rgb = np.array(mcolors.to_rgb(base_hex))
    white = np.array([1, 1, 1])
    return [mcolors.to_hex((1 - t) * white + t * rgb)
            for t in np.linspace(0.0, 1.0, n_levels)]

# %%
# ========== READ DATA ==========
df_detect = pd.read_csv(f'simulation_results/ncycle_sim_results_threshold_scaled_seed{RANDOM_SEED}.csv')
df_wide = pd.read_csv(f'simulation_results/acf_model_simulations_threshold_scaled_seed{RANDOM_SEED}.csv')

# %%
# ========== PLOT 1: PSYCHOMETRIC CURVE ==========
df_wide["Decision"] = df_wide["Decision"].astype(int)
df_avg = df_wide.groupby(["SNR", "Period"])["Decision"].mean().reset_index()


# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5),
                          )

ax = axes[0]
base = 0.45
slope = 0.30

periods_dense = np.linspace(period_min, period_max, 100)
norm_dense = np.log(periods_dense / period_min) / np.log(period_max / period_min)
crit_dense = base + norm_dense * slope

# The continuous curve
ax.plot(periods_dense, crit_dense, color='gray', linewidth=3, alpha=0.5)

# Three points for the three unit durations
for period in period_domain:
    norm_scale = np.log(period / period_min) / np.log(period_max / period_min)
    crit = base + norm_scale * slope
    ax.plot(period, crit, 'o', color=period_colors[period], markersize=20,
            markeredgecolor='black', markeredgewidth=2, zorder=3)

ax.set_xlabel('Sample Length', fontsize=25, fontweight='bold', labelpad=10)
ax.set_ylabel('Decision\nCriterion', fontsize=25, fontweight='bold', labelpad=10)
ax.tick_params(axis='both', labelsize=25)
ax.set_xticks(period_domain)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


# ========== LEFT: PSYCHOMETRIC CURVE ==========
ax = axes[1]
df_wide["Decision"] = df_wide["Decision"].astype(int)
df_avg = df_wide.groupby(["SNR", "Period"])["Decision"].mean().reset_index()

for period in period_domain:
    sub_df = df_avg[df_avg['Period'] == period]
    ax.plot(sub_df['SNR'], sub_df['Decision'],
            color=period_colors[period], linewidth=9, marker='o', markersize=15)

ax.set_xlabel('Coherence', fontsize=30, fontweight='bold')
ax.set_ylabel('P(Yes)', fontsize=30, fontweight='bold')
ax.set_yticks([0, 0.5, 1.0])
ax.tick_params(axis='both', labelsize=30)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(False)

# ========== RIGHT: CYCLES TO DETECTION ==========
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
        ax.plot(x_pair, y_pair, color=color, alpha=alpha, linewidth=9)

ax.set_xlabel('Coherence', fontsize=30, fontweight='bold')
ax.set_ylabel('N Cycles', fontsize=30, fontweight='bold')
ax.tick_params(axis='both', labelsize=30)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(False)

plt.tight_layout()
# After your first figure's tight_layout(), add invisible axes to match the colorbar footprint
plt.tight_layout()
plt.subplots_adjust(right=0.88, wspace=0.4)
# fig.savefig('plot1.png', dpi=300, bbox_inches=fixed_bbox)
plt.show()


# %%
# ========== PLOT 3: SMS — AUTOCORRELATION BY REPS ==========
df_wide["Autocorr"] = pd.to_numeric(df_wide["Autocorr"], errors='coerce')
unique_periods = sorted(df_wide["Period"].unique())
selected_periods = [unique_periods[0], unique_periods[len(unique_periods)//2],
                    unique_periods[-1]]

all_snrs = sorted(df_wide["SNR"].unique())
num_snr_levels = 10
selected_indices = np.round(np.linspace(0, len(all_snrs) - 1, num_snr_levels)).astype(int)
snrs = [all_snrs[i] for i in selected_indices]
snr_to_index = {snr: i for i, snr in enumerate(snrs)}

fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for i, period in enumerate(selected_periods):
    ax = axs[i]
    sub_df = df_wide[df_wide["Period"] == period]
    colormap = get_colormap(base_colors[period], len(snrs))

    for snr in snrs:
        df_plot = (sub_df[sub_df["SNR"] == snr]
                   .groupby("Reps")["Autocorr"].mean().reset_index())
        color = colormap[snr_to_index[snr]]
        ax.plot(df_plot["Reps"], df_plot["Autocorr"],
                color=color, linewidth=9)
        ax.scatter(df_plot["Reps"], df_plot["Autocorr"],
                   color=color, s=60, edgecolor='black', linewidth=5, alpha=0.8)

    ax.set_xticks([5, 15, 30])
    ax.set_yticks([0.2, 0.5, 0.8])
    ax.tick_params(axis='both', labelsize=30)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    ax.set_xlabel('N Cycles', fontsize=30, fontweight='bold')
    ax.set_title(period_labels[period], fontsize=30, fontweight='bold',
                 color=base_colors[period])

    if i == 0:
        ax.set_ylabel('Max Corr', fontsize=30, fontweight='bold')

# Lock positions BEFORE adding colorbars
plt.tight_layout()
plt.subplots_adjust(right=0.88, wspace=0.4)

# Add colorbars using final positions
for i, period in enumerate(selected_periods):
    ax = axs[i]
    snr_norm = mcolors.Normalize(vmin=min(snrs), vmax=max(snrs))
    box = ax.get_position()
    cbar_ax = fig.add_axes([box.x1 + 0.01, box.y0, 0.008, box.height])
    cmap_local = mcolors.LinearSegmentedColormap.from_list(
        "custom", get_colormap(base_colors[period], 256))
    cb = ColorbarBase(cbar_ax, cmap=cmap_local, norm=snr_norm, orientation='vertical')

    if i == len(selected_periods) - 1:
        cb.ax.tick_params(labelsize=20)
        cb.ax.set_yticks([0, 0.5, 1])
        cb.set_label('Coherence', fontsize=20, fontweight='bold')
    else:
        cb.ax.set_yticks([])

# fig.savefig('plot2.png', dpi=300, bbox_inches=fixed_bbox)
# plt.subplots_adjust(right=0.88, wspace=0.4)
plt.show()
# %%
