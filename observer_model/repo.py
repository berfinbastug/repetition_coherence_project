import numpy as np

def circular_autocorr(signal):
    N = len(signal)
    result = np.zeros(N)
    
    for lag in range(N):
        rolled_signal = np.roll(signal, lag)  # Shift signal cyclically
        result[lag] = np.dot(signal, rolled_signal)  # Dot product

    return result
