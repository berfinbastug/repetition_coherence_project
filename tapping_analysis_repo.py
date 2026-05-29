import numpy as np
import pandas as pd
import numpy as np
import ast
from math import pi
import matplotlib.pyplot as plt
from scipy.stats import circstd, circmean
from pycircstat2.hypothesis import rayleigh_test
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from scipy.stats import median_abs_deviation
import statsmodels.api as sm





def calculate_tap_onset_deviation(tap_onset_s, actual_onset_s):
    # Remove NaN values
    # cleaned_tap_onsets = trial_tap_onset_values[~np.isnan(trial_tap_onset_values)]   
    n_tap = len(tap_onset_s)
    # Find the index of the closest smaller value in actual_onset_time_points for each element in rt_array_trial_1
    closest_indices = np.searchsorted(actual_onset_s, tap_onset_s, side='right') - 1
    # I am doing this because there might be cases where the ntap is lower than nrep. 
    # but the code above makes them equal length. I am cutting the closest indices with respect to the
    # actual number of taps
    cut_closest_indices = closest_indices[:n_tap]
    # Identify repeated indices and store them
    # i don't know what to do with them for now (16.04.2024)
    unique_elements, counts = np.unique(cut_closest_indices, return_counts=True)
    repeated_elements = unique_elements[counts > 1]
    # Subtract tapping points from the closest smaller value in actual_onset_time_points
    deviation = tap_onset_s - actual_onset_s[cut_closest_indices]

    return deviation



# -----------------------------------------------------
# Convert delays to phase angles and unit vectors
# -----------------------------------------------------
def get_tapping_vectors(result_array, unit_duration):
    np_result_array = np.array(result_array) # result array is a list, make them numpy array
    # Subtract elements and handle remaining last elements
    # tapping_phases = (np_result_array/unit_duration) * 2 * pi
    tapping_phases = ((np_result_array / unit_duration) * 2 * pi) % (2 * pi)
    tapping_vectors = np.exp( 1j * tapping_phases )
    
    return tapping_phases, tapping_vectors



# -----------------------------------------------------
# Precomputed null distribution cache
# -----------------------------------------------------
# Simulate a null distribution of resultant length (absolute value of the mean vector) 
# values under random tapping.
_null_r_cache = {}

def get_null_r_stats(n_taps, n_permutations=1000):

    if n_taps in _null_r_cache:
        return _null_r_cache[n_taps]

    random_phases = np.random.uniform(0, 2 * np.pi, size=(n_permutations, n_taps))
    r_vals = np.abs(np.mean(np.exp(1j * random_phases), axis=1))
    mean_r = np.mean(r_vals)
    std_r = np.std(r_vals)
    _null_r_cache[n_taps] = (mean_r, std_r)
    return mean_r, std_r




def compute_z_scored_r(observed_r, n_taps):
    if n_taps < 3:
        return np.nan
    mean_r, std_r = get_null_r_stats(n_taps)
    if std_r == 0:
        return np.nan
    return (observed_r - mean_r) / std_r




# -----------------------------------------------------
# Sliding window analysis
# -----------------------------------------------------
def compute_sliding_window_values(actual_onset_s, tap_onset_s, unit_dur):
    window_size = {0.4: 4, 0.7: 7, 1.0: 10}.get(unit_dur, 4)
    step_size = {0.4: 0.4, 0.7: 0.7, 1.0: 1.0}.get(unit_dur, 4)

    norm_values = []
    z_values = []
    z_scored_r_values = []

    max_time = (max(actual_onset_s) + unit_dur) if len(actual_onset_s) > 0 else unit_dur
    window_start = 0

    while window_start + window_size <= max_time:
        window_end = window_start + window_size

        actual_window = np.array([a for a in actual_onset_s if window_start <= a < window_end])
        tap_window = np.array([t for t in tap_onset_s if window_start <= t < window_end])

        if len(actual_window) == 0 or len(tap_window) == 0:
            norm_values.append(np.nan)
            z_values.append(np.nan)
            z_scored_r_values.append(np.nan)
            
        else:
            closest_indices = np.searchsorted(actual_window, tap_window, side='right') - 1
            valid_indices = (closest_indices >= 0)
            closest_actuals = actual_window[closest_indices[valid_indices]]
            valid_tap_onsets = tap_window[valid_indices]
            tap_onset_delays = valid_tap_onsets - closest_actuals
            n_taps = len(tap_onset_delays)

            if n_taps > 0:
                tapping_phases, _ = get_tapping_vectors(tap_onset_delays, unit_dur)
                result = rayleigh_test(tapping_phases)
                norm_values.append(result.r)
                z_values.append(result.z)
                z_scored_r = compute_z_scored_r(result.r, n_taps)
                z_scored_r_values.append(z_scored_r)
            else:
                norm_values.append(np.nan)
                z_values.append(np.nan)
                z_scored_r_values.append(np.nan)

        window_start += step_size

    return norm_values, z_values, z_scored_r_values



# # within trial outlier analysis
# def analyze_trial(trial):
#     try:
#         unit_dur = trial['unit_dur']
#         all_taps = trial['tap_onset_s']
#         all_itis = np.diff(all_taps)
#         iti_indices = np.arange(len(all_itis))


#         # --- MAD Outlier Detection  ---
#         median_iti = np.median(all_itis)
#         mad_iti = median_abs_deviation(all_itis, scale='normal')
#         mad_threshold = 3 * mad_iti
#         mad_mask = np.abs(all_itis - median_iti) <= mad_threshold
#         mad_outlier_indices = np.where(~mad_mask)[0]


#         # --- Regression mask ---
#         regression_mask = np.ones(len(all_itis), dtype=bool)
#         regression_mask[mad_outlier_indices] = False

#         # --- Regression Model Fit ---
#         X_reg = iti_indices[regression_mask].reshape(-1, 1)
#         y_reg = all_itis[regression_mask]
#         reg_model = LinearRegression().fit(X_reg, y_reg)
#         predicted_itis_all = reg_model.predict(iti_indices.reshape(-1, 1))
#         lower_bound = 0.5 * predicted_itis_all
#         upper_bound = 1.5 * predicted_itis_all

#         # --- Regression Outliers ---
#         regression_outlier_indices = np.where((all_itis < lower_bound) | (all_itis > upper_bound))[0]


#         # --- All Outliers ---
#         all_outlier_iti_indices = np.unique(
#             np.concatenate([mad_outlier_indices, regression_outlier_indices])
#         )

#         # --- Clean taps & ITIs ---
#         tap_mask = np.ones(len(all_taps), dtype=bool)
#         for idx in all_outlier_iti_indices:
#             tap_mask[idx] = False
#             if idx + 1 < len(tap_mask):
#                 tap_mask[idx + 1] = False
#         cleaned_taps = all_taps[tap_mask]

#         valid_iti_mask = np.ones(len(all_itis), dtype=bool)
#         valid_iti_mask[all_outlier_iti_indices] = False
#         cleaned_itis = all_itis[valid_iti_mask]

#         # --- OLS Analysis ---
#         X_sm = sm.add_constant(iti_indices)
#         ols_model = sm.OLS(all_itis[regression_mask], X_sm[regression_mask]).fit()
#         predicted_itis_all = ols_model.predict(X_sm)
#         slope = ols_model.params[1]
#         p_value = ols_model.pvalues[1]

#         # --- Error Calculation based on valid indices only ---
#         valid_idx = np.where(regression_mask)[0]
#         start_error = predicted_itis_all[valid_idx[0]] - unit_dur
#         end_error = predicted_itis_all[valid_idx[-1]] - unit_dur

#         # --- Categorization ---
#         good_range = {
#             0.4: (0.25, 0.55),
#             0.7: (0.55, 0.85),
#             1.0: (0.85, 1.15),
#         }
#         lower, upper = good_range.get(round(unit_dur, 1), (unit_dur - 0.15, unit_dur + 0.15))
#         iti_median = np.median(all_itis)

#         if p_value >= 0.05:
#             if lower <= iti_median < upper:
#                 category = 1  # No Learning - Good Performance
#             else:
#                 category = 2  # No Learning - Bad Performance
#         else:
#             if abs(end_error) < abs(start_error):
#                 category = 3 if slope > 0 else 4  # Learning (+/-)
#             else:
#                 category = 5 if slope > 0 else 6  # Confusion (+/-)

#         return pd.Series({
#             'cleaned_itis': cleaned_itis,
#             'cleaned_taps': cleaned_taps,
#             'p_value': p_value,
#             'slope': slope,
#             'category': category,
#             'median_iti': iti_median,
#             'n_tap_filtered': len(cleaned_taps) 
#         })

#     except Exception as e:
#         print(f"Error in trial at index {trial}: {e}")
#         return pd.Series({
#             'cleaned_itis': np.nan,
#             'cleaned_taps': np.nan,
#             'p_value': np.nan,
#             'slope': np.nan,
#             'category': np.nan,
#             'median_iti': np.nan,
#             'n_tap_filtered': np.nan
#         })
    



# # within trial outlier analysis
# def analyze_trial(trial):
#     try:
#         unit_dur = trial['unit_dur']
#         all_taps = trial['tap_onset_s']
#         all_itis = np.diff(all_taps)
#         iti_indices = np.arange(len(all_itis))

#         # --- Define cycle window ---
#         cycle_start = trial['actual_onset_s'][-20]
#         cycle_end = trial['actual_onset_s'][-1] + unit_dur

#         tap_centers = all_taps[:-1] + all_itis / 2
#         last_iti_mask = (tap_centers >= cycle_start) & (tap_centers <= cycle_end)

#         # --- MAD Outlier Detection in last 15 cycles ---
#         last_itis = all_itis[last_iti_mask]
#         median_iti = np.median(last_itis)
#         mad_iti = median_abs_deviation(last_itis, scale='normal')
#         mad_threshold = 3 * mad_iti
#         mad_mask_last = np.abs(last_itis - median_iti) <= mad_threshold
#         mad_outlier_indices_last = np.where(~mad_mask_last)[0]
#         global_mad_outlier_indices = np.where(last_iti_mask)[0][mad_outlier_indices_last]

#         # --- Regression mask ---
#         regression_mask = np.ones(len(all_itis), dtype=bool)
#         regression_mask[global_mad_outlier_indices] = False

#         # --- Regression Model Fit ---
#         X_reg = iti_indices[regression_mask].reshape(-1, 1)
#         y_reg = all_itis[regression_mask]
#         reg_model = LinearRegression().fit(X_reg, y_reg)
#         predicted_itis_all = reg_model.predict(iti_indices.reshape(-1, 1))
#         lower_bound = 0.5 * predicted_itis_all
#         upper_bound = 1.5 * predicted_itis_all

#         # --- Regression Outliers in last 15 cycles ---
#         regression_outlier_indices_last = np.where(
#             last_iti_mask & ((all_itis < lower_bound) | (all_itis > upper_bound))
#         )[0]

#         # --- All Outliers ---
#         all_outlier_iti_indices = np.unique(
#             np.concatenate([global_mad_outlier_indices, regression_outlier_indices_last])
#         )

#         # --- Clean taps & ITIs ---
#         tap_mask = np.ones(len(all_taps), dtype=bool)
#         for idx in all_outlier_iti_indices:
#             tap_mask[idx] = False
#             if idx + 1 < len(tap_mask):
#                 tap_mask[idx + 1] = False
#         cleaned_taps = all_taps[tap_mask]

#         valid_iti_mask = np.ones(len(all_itis), dtype=bool)
#         valid_iti_mask[all_outlier_iti_indices] = False
#         cleaned_itis = all_itis[valid_iti_mask]

#         # --- OLS Analysis ---
#         X_sm = sm.add_constant(iti_indices)
#         ols_model = sm.OLS(all_itis[regression_mask], X_sm[regression_mask]).fit()
#         predicted_itis_all = ols_model.predict(X_sm)
#         slope = ols_model.params[1]
#         p_value = ols_model.pvalues[1]

#         # --- Error Calculation based on valid indices only ---
#         valid_idx = np.where(regression_mask)[0]
#         start_error = predicted_itis_all[valid_idx[0]] - unit_dur
#         end_error = predicted_itis_all[valid_idx[-1]] - unit_dur

#         # --- Categorization ---
#         good_range = {
#             0.4: (0.25, 0.55),
#             0.7: (0.55, 0.85),
#             1.0: (0.85, 1.15),
#         }
#         lower, upper = good_range.get(round(unit_dur, 1), (unit_dur - 0.15, unit_dur + 0.15))
#         iti_median = np.median(all_itis)

#         if p_value >= 0.05:
#             if lower <= iti_median < upper:
#                 category = 1  # No Learning - Good Performance
#             else:
#                 category = 2  # No Learning - Bad Performance
#         else:
#             if abs(end_error) < abs(start_error):
#                 category = 3 if slope > 0 else 4  # Learning (+/-)
#             else:
#                 category = 5 if slope > 0 else 6  # Confusion (+/-)

#         return pd.Series({
#             'cleaned_itis': cleaned_itis,
#             'cleaned_taps': cleaned_taps,
#             'p_value': p_value,
#             'slope': slope,
#             'category': category,
#             'median_iti': iti_median,
#             'n_tap_filtered': len(cleaned_taps) 
#         })

#     except Exception as e:
#         print(f"Error in trial at index {trial}: {e}")
#         return pd.Series({
#             'cleaned_itis': np.nan,
#             'cleaned_taps': np.nan,
#             'p_value': np.nan,
#             'slope': np.nan,
#             'category': np.nan,
#             'median_iti': np.nan,
#             'n_tap_filtered': np.nan
#         })
    





# def compute_sliding_window_values(actual_onset_s, tap_onset_s, unit_dur):
#     """
#     Computes the mean vector norm using a sliding window approach.
#     - Window size is determined by unit_dur (4s for 0.4s, 7s for 0.7s, 10s for 1s).
#     This corresponds 10 cycle
#     - Step size is always 1 cycle meaning that for 0.4 it is 0.4 seconds, for 0.7 it is 0.7 seconds and for 1 it is 1 seconds.
#     """
    
#     # Define window size based on unit duration
#     window_size = {0.4: 4, 0.7: 7, 1.0: 10}.get(unit_dur, 4)  # Default to 4s if unit_dur is unknown
#     step_size = {0.4: 0.4, 0.7: 0.7, 1.0: 1}.get(unit_dur, 4)
#     # step_size = 0.4  # Slide window every 0.2s
    

#     norm_values = []  # Store mean vector norm values, this is resultant length
#     z_values = [] 
#     p_values = []
#     median_iti_values = []
#     sd_iti_values = []
#     z_scored_r_values = []

#     # Get the total duration of the trial
#     max_time = (max(actual_onset_s) + unit_dur) if len(actual_onset_s) > 0 else unit_dur
   
#     window_start = 0
#     # Iterate through sliding windows
#     while window_start + window_size <= max_time:
#         window_end = window_start + window_size
        
#         # Get actual and tap onsets inside the current window
#         actual_window = np.array([a for a in actual_onset_s if window_start <= a < window_end])
#         tap_window = np.array([t for t in tap_onset_s if window_start <= t < window_end])

#         if len(actual_window) == 0 or len(tap_window) == 0:
#             norm_values.append(np.nan)  # No taps in this window
#         else:
#             itis = np.diff(tap_window)
#             med  = np.median(itis)
#             deviation = np.std(itis)
#             median_iti_values.append(med)
#             sd_iti_values.append(deviation)

            
#             # Find closest actual onset before each tap using np.searchsorted
#             closest_indices = np.searchsorted(actual_window, tap_window, side='right') - 1
            
#             # Ensure indices are valid (i.e., actual onset is before tap onset)
#             valid_indices = (closest_indices >= 0)
#             closest_actuals = actual_window[closest_indices[valid_indices]]
#             valid_tap_onsets = tap_window[valid_indices]

#             # Compute tap onset delays (difference between tap and actual)
#             tap_onset_delays = valid_tap_onsets - closest_actuals
            
#             n_taps = len(tap_onset_delays)
#             if n_taps > 0:
#                 # Convert to tapping vectors
#                 tapping_phases, tapping_vectors = get_tapping_vectors(tap_onset_delays, unit_dur)
#                 result = rayleigh_test(tapping_phases)
#                 p_values.append(result.pval)
#                 z_values.append(result.z)
#                 # Compute mean vector norm
#                 # mean_vector_norm = np.abs(np.mean(tapping_vectors))
#                 norm_values.append(result.r)

#                 # Z-score correction for sample size bias
#                 z_r = compute_z_scored_r(result.r, n_taps)
#                 z_scored_r_values.append(z_r)
#             else:
#                 norm_values.append(np.nan)  # No valid delays in this window
        
#         # Slide window forward by 0.2s
#         window_start += step_size

#     return norm_values, p_values, z_values, median_iti_values, sd_iti_values, z_scored_r_values # Return list of norm values for each window



# def compute_sliding_window_values(actual_onset_s, tap_onset_s, unit_dur):
#     """
#     Computes the mean vector norm using a sliding window approach.
#     - Window size is determined by unit_dur (4s for 0.4s, 7s for 0.7s, 10s for 1s).
#     This corresponds 10 cycle
#     - Step size is always 1 cycle meaning that for 0.4 it is 0.4 seconds, for 0.7 it is 0.7 seconds and for 1 it is 1 seconds.
#     """
    
#     # Define window size based on unit duration
#     window_size = {0.4: 4, 0.7: 7, 1.0: 10}.get(unit_dur, 4)  # Default to 4s if unit_dur is unknown
#     step_size = {0.4: 0.4, 0.7: 0.7, 1.0: 1}.get(unit_dur, 4)
#     # step_size = 0.4  # Slide window every 0.2s
    
#     # Get the total duration of the trial
#     max_time = (max(actual_onset_s) + unit_dur) if len(actual_onset_s) > 0 else unit_dur
#     window_start = 0
#     norm_values = []  # Store mean vector norm values, this is resultant length
#     z_values = [] 
#     p_values = []

#     # Iterate through sliding windows
#     while window_start + window_size <= max_time:
#         window_end = window_start + window_size
        
#         # Get actual and tap onsets inside the current window
#         actual_window = np.array([a for a in actual_onset_s if window_start <= a < window_end])
#         tap_window = np.array([t for t in tap_onset_s if window_start <= t < window_end])

#         if len(actual_window) == 0 or len(tap_window) == 0:
#             norm_values.append(np.nan)  # No taps in this window
#         else:
#             # Find closest actual onset before each tap using np.searchsorted
#             closest_indices = np.searchsorted(actual_window, tap_window, side='right') - 1
            
#             # Ensure indices are valid (i.e., actual onset is before tap onset)
#             valid_indices = (closest_indices >= 0)
#             closest_actuals = actual_window[closest_indices[valid_indices]]
#             valid_tap_onsets = tap_window[valid_indices]

#             # Compute tap onset delays (difference between tap and actual)
#             tap_onset_delays = valid_tap_onsets - closest_actuals
            
#             if len(tap_onset_delays) > 0:
#                 # Convert to tapping vectors
#                 tapping_phases, tapping_vectors = get_tapping_vectors(tap_onset_delays, unit_dur)
#                 result = rayleigh_test(tapping_phases)
#                 p_values.append(result.pval)
#                 z_values.append(result.z)
#                 # Compute mean vector norm
#                 # mean_vector_norm = np.abs(np.mean(tapping_vectors))
#                 norm_values.append(result.r)
#             else:
#                 norm_values.append(np.nan)  # No valid delays in this window
        
#         # Slide window forward by 0.2s
#         window_start += step_size

#     return norm_values, p_values, z_values  # Return list of norm values for each window
