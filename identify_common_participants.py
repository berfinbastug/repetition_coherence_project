# Imports and Setup
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# REPETITION DETECTION
df_det = pd.read_csv('/Users/bastugb/Desktop/repetition_coherence_project/all_dfs_merged_detection_experiment.csv')
len(df_det['participant_id'].unique())

# TAPPING
df_tap = pd.read_csv('/Users/bastugb/Desktop/repetition_coherence_project/all_participants_tap_and_actual_onset_values.csv')

df_tap['participant_id'] = df_tap['filename'].str.extract(r'pid(\d+)').astype(int)


# Find common participants
common_participants = set(df_tap['participant_id']) & set(df_det['participant_id'])
# Filter both DataFrames to only include common participants
df_tap_common = df_tap[df_tap['participant_id'].isin(common_participants)].copy()
df_det_common = df_det[df_det['participant_id'].isin(common_participants)].copy()

# Save df_tap_common
df_tap_common.to_csv('/Users/bastugb/Desktop/repetition_coherence_project/df_tap_common.csv', index=False)
# Save df_det_common
df_det_common.to_csv('/Users/bastugb/Desktop/repetition_coherence_project/df_det_common.csv', index=False)