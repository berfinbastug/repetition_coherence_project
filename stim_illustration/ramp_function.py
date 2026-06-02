# here my aim is to transform an existing matlab function into a python code
import numpy as np

# The psyramp function is designed to apply a cosine-squared ramp (fade-in and fade-out) to a signal x. 
# This can be useful in audio processing to smoothly transition the start and end of a signal to avoid 
# abrupt changes, which can create clicks or other unwanted artifacts. 

# input parameters
# x: The input signal (likely a numpy array).
# rtime: The ramp time in seconds.
# fs: The sampling frequency in Hz (samples per second).

def psyramp(x, rtime, fs):

    # calculate the ramp time array
    # rtime in seconds
    lt = len(x)  # length of the input signal
    
    # A numpy array representing the time values for the ramp duration, starting from 0 to rtime with a step of 1/fs. 
    # This array represents the time points over which the ramp effect will be applied.
    tr = np.arange(0, rtime-1/fs, 1/fs) 
    lr = len(tr)

    # adding up cosine squared ramps (up and down)
    rampup = ((np.cos(2 * np.pi * tr / rtime / 2 + np.pi) + 1) / 2) ** 2
    rampdown = ((np.cos(2 * np.pi * tr / rtime / 2) + 1) / 2) ** 2

    xr = x.copy()
    xr[:lr] = rampup * x[:lr] # Apply the ramp-up effect to the beginning of the signal.
    xr[lt - lr:lt] = rampdown * x[lt - lr:lt] # Apply the ramp-down effect to the end of the signal.
    return xr