#%%
#=====================
#IMPORT IMPORTANT STUFF
#=====================
import numpy as np
from tone_cloud_production import gencloudcoherence
import sys

# important auditory analysis toolbox given by Fede
# inside, you can find cochleagram, actually it is a cochleagram of McDermott but corrected by Fede
cochleagram_path = '/Users/bastugb/Desktop/toolboxes/TFaudition-master/pycochleagram-master'
sys.path.append(cochleagram_path)

from pycochleagram import cochleagram
import matplotlib.pyplot as plt

#%%
# =====================
#
# =====================
# Extract the relevant parameters from the current row
unitdur = 1
percentage = 0.778
nrep = 1
# create change_dict for the current row
change_dict = {'unitdur' : unitdur, 'percentage' : percentage, 'nrep' : nrep}
# Generate the sound stimulus based on the parameters
y, sP = gencloudcoherence(change_dict=change_dict)

# %%
# =====================
# COCHLEAGRAM
# =====================
# OLD
# sr = sP['fs']
# out, _, hz_cutoffs = cochleagram.human_cochleagram(signal = y, sr = 44100, n=None, low_lim=200, hi_lim=int(3500),
#                                                     sample_factor=2, padding_size=None, downsample=None, nonlinearity=None,
#                                                     fft_mode='auto', ret_mode='envs', strict=True)
# NEW
sr = sP['fs']
out, _, hz_cutoffs = cochleagram.human_cochleagram(signal = y, sr = 44100, n=None, low_lim=150, hi_lim=int(3500),
                                                    sample_factor=4, padding_size=None, downsample=None, nonlinearity=None,
                                                    fft_mode='auto', ret_mode='envs', strict=True)


# %%
# =====================
# PLOT THE COCHLEAGRAM
# =====================
# Define time axis
time_res = 1 / sr  # Because of sample_factor=2
n_time_bins = out.shape[1]
cgram_times = np.arange(n_time_bins) * time_res
cgram_freqs = hz_cutoffs

fig, ax = plt.subplots(figsize=(3, 1.1))
ax.imshow(
    out,
    aspect='auto',
    origin='lower',
    extent=[cgram_times[0], cgram_times[-1], cgram_freqs[0], cgram_freqs[-1]],
    interpolation='nearest',
    cmap='cividis'
)
ax.set_xlim(0.25, 1.2)
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
fig.savefig('cochleagram_long.svg', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
plt.show()

# %%
