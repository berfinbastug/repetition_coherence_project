Bastug et al., *"Computational Principles of Auditory Object Formation Are Revealed by Repetition Coherence."*

# Interactive Demo
[▶ Explore the stimuli](https://berfinbastug.github.io/repetition_coherence_project/)   
Adjust coherence and unit duration, hear the tone clouds.

---

# Preprocessing & Utilities
These scripts handle data preparation and minor supplementary outputs. They are not tied to a specific manuscript figure but feed into the main analysis pipelines. 

### `identify_common_participants.py`  
Reads the separate CSVs for the two tasks (detection and SMS), identifies participants present in both, filters each data frame to that common set, and writes the results. 

Inputs:
- `all_dfs_merged_detection_experiment.csv` 
- `all_participants_tap_and_actual_onset_values.csv` 

Outputs:
- `df_det_common.csv` 
- `df_tap_common.csv` 

---

### `circular_tap_distributions.py`  
Example circular histograms (Figure 5B inset). 
Produces polar histograms of tapping-phase distributions for one good and one bad trial examples.  

---

# Repetition Detection — Analysis & Figure Code
Analysis and figure-generation scripts for the **repetition detection task**  

### `rep_det_figures_pyes.py`  
Detection accuracy (Figure 2)
Computes proportion of "yes" responses as a function of repetition coherence and unit duration, fits psychometric functions, and produces:

- **Figure 2A** — P(yes) vs. coherence, collapsed across unit durations (Weibull fit).
- **Figure 2B** — P(yes) vs. coherence, separately for each unit duration (0.4, 0.7, 1.0 s), with per-condition Weibull fits.
- **Figure 2C** — Violin plot of individual 50% detection thresholds per unit duration.

Key steps:
1. Load `df_det_common.csv` (trial-level detection data).
2. Compute `n_cycle = rt / unitdur`; exclude anticipatory trials (`n_cycle < 1`).
3. Remove RT outliers per (coherence × unit duration) cell using the IQR rule.
4. Remove blocks with low accuracy (|z| > 3 on block-level accuracy).
5. Fit a 4-parameter Weibull (α threshold, β slope, γ guess, λ lapse) by maximum-likelihood (binomial NLL, Powell optimizer) at the group level and per participant.
6. Extract individual 50%-yes thresholds for the violin plot.

Outputs:
- `detection_performance_for_r.csv` — per-participant condition summaries for downstream R analyses (ANOVA).
- `threshold_50_detection_for_r.csv` — individual detection thresholds.
- `weibull_combined.svg`, `threshold_violin.svg` — figure panels.

---

### `rep_det_figures_rt.py` 
Reaction times & cycles (Figure 3) 
Computes how long participants needed to reach a detection decision, in both seconds and repetition cycles.

- **Figure 3A** — Mean RT (s) vs. coherence, per unit duration, with line opacity scaled to accuracy.
- **Figure 3B** — Mean number of cycles to detection vs. coherence, per unit duration.

Key steps:
1. Load the same `df_det_common.csv` and apply the same outlier pipeline as above (anticipatory + IQR + bad-block exclusion).
2. Restrict to **correct trials** for RT/N-cycles summaries.
3. Per (participant × coherence × unit duration), compute median RT, median N cycles, and MADs.
4. Plot group-level means with SEM bands; line opacity encodes mean accuracy in each segment.

Outputs:
- `n_cycle_for_r.csv` — per-participant summaries for R-side stats.
- `ncycle_rt_combined.svg` — combined RT + N-cycles figure.

---

### Input data

Both scripts read the same file:

```
/Users/bastugb/Desktop/repetition_coherence_project/df_det_common.csv
```

Expected columns:
- `participant_id`
- `block_idx`
- `percentage` — repetition coherence (0–1)
- `unitdur` — unit duration in seconds (0.4, 0.7, 1.0)
- `rt` — reaction time from sequence onset (s)
- `correct` — 1/0
- `actual_response` — 1 = "yes", 0 = "no"

---

### Outlier pipeline (shared by both scripts)

1. Exclude anticipatory trials: `n_cycle = rt / unitdur < 1`.
2. RT outliers per (coherence × unit duration) cell using Tukey's 1.5 × IQR rule.
3. Exclude blocks with |z| > 3 on block-level accuracy.

Total excluded: ~4.05% of trials (manuscript Methods).

---

### Notes / TODO

- [ ] Replace hard-coded absolute paths with a config or CLI argument.
- [ ] Factor shared outlier-cleaning code into a small `utils.py` (currently duplicated across the two scripts).
- [ ] Move figure output to a dedicated `figures/` directory.

---

# Sensorimotor Synchronization — Analysis & Figure Code
Analysis and figure-generation scripts for the **sensorimotor synchronization (SMS) task**  
These scripts process trial-level tapping data and produce the panels in Figures 4 and 5 of the manuscript. 

### `tapping_analysis_repo.py` 
This is a shared helper module

`get_tapping_vectors(delays, unit_dur)` — convert tap-onset delays (s) to phase angles (rad) and unit vectors on the complex plane.
`get_null_r_stats(n_taps, n_permutations = 1000)` — Monte-Carlo null distribution of the mean resultant length under uniformly distributed random phases, cached per `n_taps`.
`compute_z_scored_r(observed_r, n_taps)` — z-score of an observed resultant length against the null distribution. This was used as the tappinng-phase consistency measure throughout. 
`compute_sliding_window_values(actual_onset_s, tap_onset_s, unit_dur)` — sliding-window analysis. Window = 10 cycles, step = 1 cycle. Within each window: aligns taps to nearest preceding onset, computes resultant length, Rayleigh z, and z-scored r. Returns three lists (one value per window).

---

### `tapping_figures_sync.py` 
Synchronization success and threshold (Figure 4 + Figure 5B-C) 
End-to-end pipeline for the binary "successful synchronization" outcome and the within-trial stable-phase analysis. 

Key steps: 
1. Load `df_tap_common.csv`. 
2. Parse `actual_onset_values`, `tap_onset_values` strings into arrays; convert sample indices to seconds (sample_rate = 44100). 
3. Drop trials with ≤ 2 taps. Remove `n_tap` outliers per (coherence x unit duration) cell using Tukey's 1.5 × IQR. 
4. Last 20-cycles analysis (stable phase): 
   1. extract taps in the final 20 cycles, compute tap-onset delays, convert to phases. 
   2. compute resultant length per trial --> z-score against null --> significance flag (z > z_{0.05}). 
5. Aggregate per (participant x coherence x unit duration) and fit per-participant Weibull psychometric functions (MLE / Binomial NLL, Powell optimized) to the proportion of significant trials. 
6. All-cycles analysis (Figure 5B-C, restricted to coherence = 1): 
   1. recompute phases over the entire trial. 
   2. `locate_within_range_taps(...)` defines the stable-phase range as circular mean ± circular SD of the last 20 cycles, then finds the earliest tap that falls inside the range and is followed by ≥ 5 consecutive inside taps (`min_stable_index`). 

Outputs:
- **Figure 4A** — P(Sync) vs. coherence, collapsed across unit durations 
- **Figure 4B** — P(Sync) vs. coherence per unit duration 
- **Figure 4C** — Violin of individual 50% sync thresholds per unit duration 
- **Figure 5B** — Example trials (good and bad) showing tap phase vs tap index with stable-phase band 
- **Figure 5C** - Violin. of `min_stable_index` (earliest stable tap) per unit duration at coherence = 1  

---

### `sms_figures_exp.py` 
Within trial dynamics (Figure 5A). 
Sliding-window analysis of tapping phase consistency and exponential-saturation fits. 

Key steps: 
1. Identical load + outlier-cleaning steps to `sms_figures_sync.py`(basically the first 110 lines are the exact duplicates, at some point create a shared `load_tapping_data()` helper). 
2. Per trial, call `compute_sliding_window_values` from `tapping_analysis_repo` --> `z_scored_r` time series across sliding windows. 
3. Pad trials to a common length and average per (participant x coherence x unit duration), then again per (coherence x unit duration). 
4. Fit `f(x) = a·(1 − exp(−b·x)) + c` per participant x condition (with scipy curve_fit). 
5. Aggregate fits to group level. 

Outputs:
- **Figure 5A** — Three panels (one per unit duration) showing z-scored resultant length as a function of sliding window index. Coherence is represented as a color gradient. There is a significance line (p = 0.05). 
  
---

### Input data

Both SMS scripts read the same file:

```
/Users/bastugb/Desktop/repetition_coherence_project/df_tap_common.csv
```

Expected columns:
- `participant_id`
- `filename` (contains `block_<n>_itrial_<n>)
- `percentage` — repetition coherence (0–1)
- `unit_dur` — unit duration in seconds (0.4, 0.7, 1.0)
- `actual_onset_values` — [AT SOME POINT ADD DETAILS OF WHAT THEY ARE]
- `tap_onset_values` — [AT SOME POINT ADD DETAILS OF WHAT THEY ARE]

---

### Notes / TODO

- [ ] Lines ~1-110 of the two SMS scripts are identical. Extract into `load_tapping_data()` in `tapping_analysis_repo.py`. 
- [ ] Hard coded absolute path.  
- [ ] In `sms_figures_sync.py` the `tap_onset_delays_list` / "select taps in window" block is itself duplicated twice (last-20-cycles version and all-cycles version) — pull out a `delays_in_window(actual, taps, window)` helper.

---

# Stimulus Generation Schematics 
Figure 1. All these scripts are inside stim_illustration. 

### `stimulus_params.py` 
Default parameter dictionary (`sP_default`): frequency range (200-3000 Hz), 0.4-octave grid step, 0.05 s time step, tone duration, unit duration, nrep, coherence, sample rate (44100 Hz), ramp time. 

---

### `ramp_function.py` 
`psyramp(x, rtime, fs)`: applies cosine-squared onset/offset ramps to individual tone pips (0.025s default). Prevents clicks at tone boundaries.

---

### `tone_cloud_production.py` 
`gencloudcoherence(sP, change_dict)`: the main stimulus factory. Builds the time-frequency grid, selects frozen vs. new tones according to `percentage`, tiles across `nrep` cycles, synthesizes the waveform (summed ramped sinusoids), normalizes, zero-pads. Returns `(waveform, params)`

---

### `single_tone_cloud_schematic.py` 
One unit of a tone cloud (1 cycle, 0.4 s). Produces schematic tone clouds` --> Figure A (spectogram-style grid with horizontal lines for each tone pip). 

---

### `repetition_coherence_gradient_schematic.py` 
2-cycle strips at five coherence levels (0, 0.22, 0.44, 0.67, 1), side by side. Blue = frozen, gray = new. --> Figure 1C.  

---

### `task_illustration_schematic.py` 
4-cycle "yes" trial (coherence = 0.78) and "no" trial (coherence =0), plotted separately --> `4rep_trial_yes.svg`, `4rep_trial_no.svg`--> Figure 1D. 

---

### `tap_illustrations.py` 
Synthetic damped-sinusoid tap waveform (4 taps with jitter and negative asynchrony) --> `sms_taps_jittered.svg` --> Figure 1E tap overlay. 

---

### `plot_cochleagram.py` 
Generates a tone cloud via `gencloudcoherence`, computes a cochleagram (using `pycochleagram`) and plots it --> Figure 1B. Requires the external  `pycochleagram` library and a local path to the toolbox. 
