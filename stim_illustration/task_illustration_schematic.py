import numpy as np
import copy
import matplotlib.pyplot as plt

# %%
UNITDUR = 0.4
NREP = 4
SEED = 42

# draw grids of various percentage values
PERC_VALS = [0.778, 0]
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = '0.8'

# =====================
#
# =====================
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

# USEFUL FOR FIGURES
# Extend the time axis to include repetitions
total_duration = NREP * UNITDUR
times_extended = np.arange(0, total_duration + timestep, timestep)
important_times = np.arange(UNITDUR, NREP * UNITDUR, UNITDUR)

# %%
# ===========================
# FIRST GENERATE A "YES" TRIAL
# ===========================
# deal with the repeating percentage of the tones
p = PERC_VALS[0]

if (p == 0):
    nreptones = 0
    nnewtones = ntones - nreptones
elif (p == 1):
    nreptones = ntones
    nnewtones = ntones - nreptones
else:
    nreptones = int(np.ceil(ntones * p))
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
# ===========================
# PLOT A "YES" TRIAL
# ===========================
fig, ax1 = plt.subplots(figsize=(3, 1.2), sharex=True)
# Add vertical reference lines for time
for time in times_extended:
    is_important_time = False  # Flag to track if the current time is 'important'
    for important_time in important_times:
        if np.isclose(time, important_time, rtol=1e-9):  # Use np.isclose for floating-point comparison
            is_important_time = True
            break  # No need to check other important times if a match is found
    
    if is_important_time:
        ax1.axvline(x=time, color='gray', linestyle='--', linewidth=2, alpha=0.7)
    else:
        ax1.axvline(x=time, color='gray', linestyle='--', linewidth=0.3, alpha=0.3)

# Add horizontal reference lines for frequency
for freq in freqgrid_plot:
    ax1.axhline(y=freq, color='gray', linestyle='--', linewidth=0.3, alpha=0.3)

# Plot repeated tones
for rep in range(NREP):
    for i, j in zip(r_rep, c_rep):
        start_time = bigzt[i, j + rep * ntsteps]
        end_time = start_time + 0.05
        freq = bigzf[i, j + rep * ntsteps]
        ax1.plot([start_time, end_time], [freq, freq], color = "#0A73BE", linewidth = 1.5)

# Plot non-repeated tones
for rep in range(NREP):
    for i, j in zip(r_new, c_new):
        start_time = bigzt[i, j + rep * ntsteps]
        end_time = start_time + 0.05
        freq = bigzf[i, j + rep * ntsteps]
        ax1.plot([start_time, end_time], [freq, freq], color="#8A8A8AB6", linewidth=1.5)


ax1.set_xlim(0, total_duration + 0.01)
ax1.set_ylim(lowf, highf + 500)
ax1.set_xlabel("")
ax1.set_ylabel("")

# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
ax1.set_xticks([])
ax1.set_yticks([])


plt.tight_layout()
fig.savefig('4rep_trial_yes.svg', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
plt.show()



# %%
# =====================
#
# =====================
p = PERC_VALS[1]

if (p == 0):
    nreptones = 0
    nnewtones = ntones - nreptones
elif (p == 1):
    nreptones = ntones
    nnewtones = ntones - nreptones
else:
    nreptones = int(np.ceil(ntones * p))
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


fig, ax2 = plt.subplots(figsize=(3, 1.2), sharex=True)
# Add vertical reference lines for time
for time in times_extended:
    is_important_time = False  # Flag to track if the current time is 'important'
    for important_time in important_times:
        if np.isclose(time, important_time, rtol=1e-9):  # Use np.isclose for floating-point comparison
            is_important_time = True
            break  # No need to check other important times if a match is found
    
    if is_important_time:
        ax2.axvline(x=time, color='gray', linestyle='--', linewidth=2, alpha=0.7)
    else:
        ax2.axvline(x=time, color='gray', linestyle='--', linewidth=0.3, alpha=0.3)

# Add horizontal reference lines for frequency
for freq in freqgrid_plot:
    ax2.axhline(y=freq, color='gray', linestyle='--', linewidth=0.3, alpha=0.3)

# Plot repeated tones
for rep in range(NREP):
    for i, j in zip(r_rep, c_rep):
        start_time = bigzt[i, j + rep * ntsteps]
        end_time = start_time + 0.05
        freq = bigzf[i, j + rep * ntsteps]
        ax2.plot([start_time, end_time], [freq, freq], color = "#0A73BE", linewidth = 1.5)

# Plot non-repeated tones
for rep in range(NREP):
    for i, j in zip(r_new, c_new):
        start_time = bigzt[i, j + rep * ntsteps]
        end_time = start_time + 0.05
        freq = bigzf[i, j + rep * ntsteps]
        ax2.plot([start_time, end_time], [freq, freq], color="#8A8A8AB6", linewidth=1.5)


ax2.set_xlim(0, total_duration + 0.01)
ax2.set_ylim(lowf, highf + 500)
ax2.set_xlabel("")
ax2.set_ylabel("")

# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
ax2.set_xticks([])
ax2.set_yticks([])


plt.tight_layout()
fig.savefig('4rep_trial_no.svg', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
plt.show()
# %%
