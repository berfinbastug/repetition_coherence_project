import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast

# --- Setup ---
base_stimuli_dir = 'stimuli_tapping'  # top-level stimuli folder
df = pd.read_csv('all_participants_tap_and_actual_onset_values.csv')  # or load your DataFrame


df_tap_common = df
df_tap_common['block_idx'] = df_tap_common['filename'].str.extract(r'block_(\d+)').astype(int)
df_tap_common['trial_idx'] = df_tap_common['filename'].str.extract(r'itrial_(\d+)').astype(int)
df_tap_common['actual_onset_values'] = df_tap_common['actual_onset_values'].apply(lambda x: ast.literal_eval(x))
df_tap_common['tap_onset_values'] = df_tap_common['tap_onset_values'].apply(lambda x: ast.literal_eval(x))
df_tap_common['percentage'] = df_tap_common['percentage']
df_tap_common['n_tap'] = df_tap_common['tap_onset_values'].apply(len) # Count number of taps


# Convert tap_onset_values (assumed list/array of sample indices) to seconds
sample_rate = 44100
df_tap_common['tap_onset_s'] = df_tap_common['tap_onset_values'].apply(lambda taps: np.array(taps) / sample_rate)
df_tap_common['actual_onset_s'] = df_tap_common['actual_onset_values'].apply(lambda onsets: np.array(onsets) / sample_rate)

df_tap_filtered = df_tap_common[(df_tap_common['percentage'] == 1.0) & (df_tap_common['unit_dur'] == 0.4 )]
# --- Example: For one row/trial in the participant data ---
row = df_tap_filtered.iloc[3]

# --- Parse needed parameters from filename ---
# filename = row['file_name']
# unit_dur = float(filename.split('unitdur_')[1].split('_')[0])
# percentage = float(filename.split('percentage_')[1].split('_')[0])

# # Optional: infer block number if needed from filename
# block_num = int(filename.split('block_')[1].split('_')[0])

block_num = int(row['block_idx'])
# --- Construct expected stimulus folder and match file ---
stim_folder = f'tapping_experiment_block{block_num}'
stim_dir = os.path.join(base_stimuli_dir, stim_folder)

# Tolerance for float comparison (due to rounding issues)
def match_float(value, target, tol=1e-6):
    return abs(value - target) < tol


unit_dur = row['unit_dur']
percentage = row['percentage']
# Find matching stimulus file
matching_file = None
for fname in os.listdir(stim_dir):
    if fname.endswith('.wav') and 'unitdur' in fname and 'percentage' in fname:
        u = float(fname.split('unitdur_')[1].split('_')[0])
        p = float(fname.split('percentage_')[1].split('.wav')[0])
        if match_float(u, unit_dur) and match_float(p, percentage):
            matching_file = fname
            break


if matching_file is None:
    raise FileNotFoundError("Matching stimulus file not found.")

# %%
stim_path = os.path.join(stim_dir, matching_file)

# --- Load the audio stimulus ---
y, sr = librosa.load(stim_path)

# cut the signal
# --- Extract tapping points ---
# tap_times = eval(row['tap_onset_s']) if isinstance(row['tap_onset_s'], str) else row['tap_onset_s']

tap_times = row['tap_onset_s']
actual_onset_first = row['actual_onset_values'][0]
onset_fifth = row['actual_onset_values'][4]

signal_segment = y[actual_onset_first:onset_fifth]
#%%
# --- Compute amplitude envelope (RMS) ---
frame_length = 8000
hop_length = 512
rms = librosa.feature.rms(y=signal_segment, frame_length=frame_length, hop_length=hop_length)[0]
rms_times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop_length)


# %%




# --- Plot waveform and taps ---
time_axis = np.linspace(0, len(y) / sr, num=len(y))
plt.figure(figsize=(10, 4))
# plt.plot(time_axis, y, alpha=0.3, label='Waveform')
plt.plot(rms_times, rms, color='black', label='RMS Envelope')
# plt.vlines(tap_times[0:4], ymin=0, ymax=np.max(rms), color='gray', linestyle='--', label='Taps')
# plt.vlines(actual_onset_times, ymin=0, ymax=np.max(rms), color='blue', linestyle='--', label='Onsets')
plt.ylim([0.05, 0.062])

plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Stimulus waveform and tap responses')
plt.legend()
plt.tight_layout()
plt.show()

# %%
