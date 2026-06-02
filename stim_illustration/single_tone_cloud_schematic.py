import numpy as np
import copy
import stimulus_params
import matplotlib.pyplot as plt

# %%
UNITDUR = 0.4
PERCENTAGE = 1
NREP = 1
SEED = 42
# -------------------------------------------------

np.random.seed(SEED)
lowf, highf, fstep = 200, 3000, 0.4
timestep = 0.05

# compute edges of the frequency grid
freqgrid = [lowf]
while freqgrid[-1] * 2**fstep <= highf:
    freqgrid.append(freqgrid[-1] * 2**fstep)
freqgrid = np.array(freqgrid)
nfsteps = len(freqgrid)

# create a time grid for one repeat
timegrid = np.arange(0, UNITDUR-timestep + 1e-5, timestep)

# for plotting purposes I should add the end point 
# of the time and frequency grid 
timegrid_plot = np.append(timegrid, UNITDUR)
upper_limit = highf + 500
freqgrid_plot = np.append(np.array(freqgrid), upper_limit)

# initialization
nfsteps = len(freqgrid)
ntsteps = len(timegrid)

# create random perturbations for frequency and time
fnorm = np.random.rand(nfsteps, ntsteps)  # [0, 1)
tnorm = np.random.rand(nfsteps, ntsteps)  # [0, 1)

# now build the actual frequency and time matrices
bigf = np.tile(np.array(freqgrid).reshape(-1, 1), (1, ntsteps))  # nominal values
zf = 2 ** (np.log2(bigf) + fnorm * fstep)  # perturbed values
bigt = np.tile(timegrid, (nfsteps, 1))  # nominal values
zt = bigt + tnorm * timestep  # perturbed values

# just to plot the single unit grid, i need to generate new tile
bigf_plot = np.tile(freqgrid_plot.reshape(-1,1), (1, len(timegrid_plot)))
bigt_plot = np.tile(timegrid_plot, (len(freqgrid_plot), 1))

ntones = bigf.size

if (PERCENTAGE == 0):
    nreptones = 0
    nnewtones = ntones - nreptones
elif (PERCENTAGE == 1):
    nreptones = ntones
    nnewtones = ntones - nreptones
else:
    nreptones = int(np.ceil(ntones * PERCENTAGE))
    nnewtones = ntones - nreptones

# who are the lucky few, select repeated and new tones
idxdraw = np.random.permutation(ntones)
idxreptones = idxdraw[:nreptones]
idxnewtones = idxdraw[nreptones:]

r_rep, c_rep = np.unravel_index(idxreptones, (nfsteps, ntsteps), 'C')
r_new, c_new = np.unravel_index(idxnewtones, (nfsteps, ntsteps), 'C')

bigzf = np.empty((nfsteps, 0))  # Initialize an empty array with the appropriate shape
bigzt = np.empty((nfsteps, 0))  # Initialize an empty array with the appropriate shape

for idelay in range(1, NREP + 1):
    
    # create new frequency and time matrices
    newzf = zf.copy()  # create a copy of the frequency matrix
    newzt = zt.copy()

    # create new perturbation matrices
    newfnorm = np.random.rand(nfsteps, ntsteps)
    newtnorm = np.random.rand(nfsteps, ntsteps)

    # repeat the matrices
    zf[r_new, c_new] = 2 ** (np.log2(bigf[r_new, c_new]) + newfnorm[r_new, c_new] * fstep)
    zt[r_new, c_new] = bigt[r_new, c_new] + newtnorm[r_new, c_new] * timestep
    bigzf = np.concatenate((bigzf, zf), axis = 1)

    x = zt + (idelay - 1) * UNITDUR
    bigzt = np.concatenate((bigzt, x), axis = 1)



# %%
FIGSIZE = (2, 1.1) 
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = '1.1'


fig, ax = plt.subplots(figsize=FIGSIZE)

# Extend the time axis to include repetitions
total_duration = NREP * UNITDUR
times_extended = np.arange(0, total_duration + timestep, timestep)

# Define the specific time values where you want thick lines

# Add vertical reference lines for time
for time in times_extended:
    ax.axvline(x=time, color='gray', linestyle='--', linewidth=0.3, alpha=0.4)

# Add horizontal reference lines for frequency
for freq in freqgrid_plot:
    ax.axhline(y=freq, color='gray', linestyle='--', linewidth=0.3, alpha=0.4)

# Plot repeated tones
for rep in range(NREP):
    for i, j in zip(r_rep, c_rep):
        start_time = bigzt[i, j + rep * ntsteps]
        end_time = start_time + 0.05
        freq = bigzf[i, j + rep * ntsteps]
        ax.plot([start_time, end_time], [freq, freq], color="#0A73BE", linewidth=1.5)

# Plot non-repeated tones
for rep in range(NREP):
    for i, j in zip(r_new, c_new):
        start_time = bigzt[i, j + rep * ntsteps]
        end_time = start_time + 0.05
        freq = bigzf[i, j + rep * ntsteps]
        ax.plot([start_time, end_time], [freq, freq], color='black', alpha=0.2, linewidth=2)


ax.set_xlim(0, total_duration + 0.01)
ax.set_ylim(lowf, highf + 500)
ax.set_xlabel("")
ax.set_ylabel("")

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
# ax.set_yticklabels(['100', '500', '1000', '2000', '3000'])

# ax.tick_params(axis='both', labelsize=12)
fig.savefig('schematic_tc_short.svg', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
plt.show()

# %%
