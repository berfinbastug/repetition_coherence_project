import numpy as np
import numpy as np
from math import pi
from pycircstat2.hypothesis import rayleigh_test






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



