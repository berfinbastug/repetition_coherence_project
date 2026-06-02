import numpy as np
import matplotlib.pyplot as plt

# ===========================
# PARAMETERS — tweak these
# ===========================
N_CYCLES = 4                # how many taps
JITTER_SD = 0.07           # asynchrony jitter (s); set 0 for perfectly periodic

# Damped-wave shape
WAVE_FREQ = 35              # oscillation frequency (Hz)
DAMPING = 40                # higher = faster decay (try 20-60)
WAVE_DURATION = 0.18        # how long the wave is computed for
WAVE_AMPLITUDE = 1.4

SAMPLING_RATE = 5000        # higher = smoother curves
LINE_COLOR = '#222222'      # near-black, softer than pure black
LINE_WIDTH = 1.2

NEG_ASYNCHRONY = 0.02       # taps fall ~20ms before each beat (Repp 2005)
                            # set to 0 to disable

TAP_START_OFFSET = 0.15      # silent buffer before first tap (seconds)
TOTAL_DURATION = 1.6 + TAP_START_OFFSET

# ===========================
# COMPUTE TAP TIMES
# ===========================
rng = np.random.default_rng(7)
cycle_dur = TOTAL_DURATION / N_CYCLES
click_times = []
for i in range(N_CYCLES):
    beat = TAP_START_OFFSET + i * cycle_dur
    # beat = i * cycle_dur
    jitter = rng.normal(0, JITTER_SD)
    click_times.append(beat + jitter - NEG_ASYNCHRONY)

# ===========================
# BUILD WAVE SIGNAL
# ===========================
t = np.linspace(0, TOTAL_DURATION + 0.01, int((TOTAL_DURATION + 0.01) * SAMPLING_RATE))
y = np.zeros_like(t)

for click in click_times:
    idx_start = np.searchsorted(t, click)
    t_wave = np.linspace(0, WAVE_DURATION, int(WAVE_DURATION * SAMPLING_RATE))
    # damped sinusoid with a soft attack (kills the initial downward swing)
    attack = 1 - np.exp(-200 * t_wave)
    wave = WAVE_AMPLITUDE * attack * np.exp(-DAMPING * t_wave) \
           * np.sin(2 * np.pi * WAVE_FREQ * t_wave)
    idx_end = min(idx_start + len(wave), len(y))
    if idx_start >= 0:
        y[idx_start:idx_end] += wave[:idx_end - idx_start]

# %%
# ===========================
# PLOT
# ===========================
fig, ax = plt.subplots(figsize=(3, 0.5))
ax.plot(t, y, color=LINE_COLOR, linewidth=LINE_WIDTH, solid_capstyle='round')

ax.set_xlim(0, TOTAL_DURATION + 0.01)
ax.set_ylim(-1.0, 1.0)
ax.set_xticks([])
ax.set_yticks([])
for s in ['top', 'right', 'bottom', 'left']:
    ax.spines[s].set_visible(False)

plt.tight_layout()
fig.savefig('sms_taps_jittered.svg', dpi=300, transparent=True,
             bbox_inches='tight', pad_inches=0)
plt.show()

