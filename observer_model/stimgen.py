import numpy as np
# import scipy.stats as st

# early version of stimulus generation for detection experiment
def generate_stimulus(features, prop=1.0, repeat=False, length=100, pad=True):
    
    noiseA = np.random.normal(loc=0.0, scale=0.2, size=[1, length])
    noiseA = noiseA / np.linalg.norm(noiseA)

    noiseB = np.random.normal(loc=0.0, scale=0.2, size=[1, length]) 
    noiseB = noiseB / np.linalg.norm(noiseB)

    # choose feature index
    feature_choiceA = np.random.randint(0, features.shape[0])
    feature_choiceB = feature_choiceA
    if not repeat:
        while feature_choiceB == feature_choiceA:
            feature_choiceB = np.random.randint(0, features.shape[0])
            

    sigA = features[np.newaxis, feature_choiceA, :]
    sigA = sigA / np.linalg.norm(sigA)
    # choose time position of feature (randomly)
    max_choiceA = noiseA.shape[1] - sigA.shape[1]
    time_choiceA = np.random.randint(0, max_choiceA) if max_choiceA > 0 else 0
    sigA = np.concat([np.zeros([1, time_choiceA]), sigA, 
                      np.zeros([1, noiseA.shape[1] - time_choiceA - sigA.shape[1] ]) ], 
                      axis=1)
    
    time_choiceB = time_choiceA
    sigB = features[np.newaxis, feature_choiceB, :]
    sigB = sigB / np.linalg.norm(sigB)
    max_choiceB = noiseB.shape[1] - sigB.shape[1]
    if (not repeat) and (max_choiceB > 0):
        while time_choiceA == time_choiceB:
            time_choiceB = np.random.randint(0, max_choiceB) if max_choiceB > 0 else 0

    sigB = np.concat([np.zeros([1, time_choiceB]), sigB, 
                      np.zeros([1, noiseB.shape[1]-time_choiceB-sigB.shape[1]])], 
                      axis=1)

    # normalize signals
    sigA = sigA / np.linalg.norm(sigA)
    sigB = sigB / np.linalg.norm(sigB)
    
    # make normalized stimuli based on signals and noise
    stimA = prop * sigA + (1 - prop) * noiseA
    stimA = stimA / np.linalg.norm(stimA)
    stimB = prop * sigB + (1 - prop) * noiseB
    stimB = stimB / np.linalg.norm(stimB)

    if pad:  # add zero padding 
        len_pad = (200 - length) // 2
        # pads before and after with len_pad
        stimA = np.pad(stimA, ((0, 0), (len_pad, len_pad)))
        stimB = np.pad(stimB, ((0, 0), (len_pad, len_pad)))

    return stimA, stimB


# early version of stimulus generation for the tapping simulations
def make_stimulus(features, prop=1.0, repeat=False, length=100, pad=False, num=5):
    
    len_pad = (200 - length) // 2

    noiseA = np.random.normal(loc=0.0, scale=0.2, size=[1, length])
    noiseA = noiseA / np.linalg.norm(noiseA)

    # choose feature index
    feature_choiceA = np.random.randint(0, features.shape[0])

    # Make first signal
    sigA = features[np.newaxis, feature_choiceA, :]
    sigA = sigA / np.linalg.norm(sigA)
    # choose time position of feature (randomly)
    max_choiceA = noiseA.shape[1] - sigA.shape[1]
    time_choiceA = np.random.randint(0, max_choiceA) if max_choiceA > 0 else 0
    sigA = np.concat([np.zeros([1, time_choiceA]), sigA, 
                      np.zeros([1, noiseA.shape[1] - time_choiceA - sigA.shape[1] ]) ], 
                      axis=1)
    sigA = sigA / np.linalg.norm(sigA)

    stimA = prop * sigA + (1 - prop) * noiseA
    stimA = stimA / np.linalg.norm(stimA)
    stimA = np.pad(stimA, ((0, 0), (len_pad, len_pad))) if pad else stimA
    stimuli = [stimA]

    # generate the rest of the chunks 

    for _ in range(0, num - 1):  

        noiseB = np.random.normal(loc=0.0, scale=0.2, size=[1, length]) 
        noiseB = noiseB / np.linalg.norm(noiseB)

        feature_choiceB = feature_choiceA
        if not repeat:
            while feature_choiceB == feature_choiceA:
                feature_choiceB = np.random.randint(0, features.shape[0])
            
        sigB = features[np.newaxis, feature_choiceB, :]
        max_choiceB = noiseB.shape[1] - sigB.shape[1]

        time_choiceB = time_choiceA
        if (not repeat) and (max_choiceB > 0):
            while time_choiceA == time_choiceB:
                time_choiceB = np.random.randint(0, max_choiceB) if max_choiceB > 0 else 0

        sigB = np.concat([np.zeros([1, time_choiceB]), sigB, 
                          np.zeros([1, noiseB.shape[1] - time_choiceB - sigB.shape[1] ]) ], 
                          axis=1)
        sigB = sigB / np.linalg.norm(sigB)
        stimB = prop * sigB + (1 - prop) * noiseB
        stimB = stimB / np.linalg.norm(stimB)
        stimB = np.pad(stimB, ((0, 0), (len_pad, len_pad))) if pad else stimB
        
        stimuli.append(stimB)

    return stimuli


def make_julez_stream(snr=1.0, len_unit=10,  num_reps=2):
    # len_stream = num_reps * len_unit
    sig_unit = np.random.uniform(-1.0, 1.0, size=[len_unit])
    signal = np.tile(sig_unit, reps=[num_reps])
    signal -= signal.mean()
    # signal /= np.linalg.norm(signal)
    signal /= rms(signal)
    
    noise = np.random.uniform(-1.0, 1.0, size=[signal.shape[0]])
    noise -= noise.mean()
    # noise /= np.linalg.norm(noise)
    noise /= rms(noise)

    stream = (snr * signal) + ( (1 - snr) * noise)
    stream -= stream.mean()
    # stream /= np.linalg.norm(noise)
    stream /= rms(stream)

    return stream




def make_julez_stream_with_jitter(snr=1.0, len_unit=10, num_reps=2, temporal_uncertainty_std=0):
    sig_unit = np.random.uniform(-1.0, 1.0, size=[len_unit])
    signal_stream = np.zeros(len_unit * num_reps)
    
    for i in range(num_reps):
        jitter = int(np.round(np.random.normal(0, temporal_uncertainty_std)))
        onset = i * len_unit + jitter
        onset = np.clip(onset, 0, len(signal_stream) - len_unit)
        signal_stream[onset:onset + len_unit] += sig_unit
    
    signal_stream -= signal_stream.mean()
    signal_stream /= rms(signal_stream)
    
    noise = np.random.uniform(-1.0, 1.0, size=signal_stream.shape)
    noise -= noise.mean()
    noise /= rms(noise)

    stimulus = snr * signal_stream + (1 - snr) * noise
    stimulus -= stimulus.mean()
    stimulus /= rms(stimulus)
    
    return stimulus


def rms(signal):
    return np.sqrt(np.mean(np.abs(signal) ** 2))

# def make_signal(feature, time_choice):
#     sigA = np.concat([np.zeros([1, time_choice]), sigA, 
#                     np.zeros([1, noise.shape[0]-time_choice])])
#     return signal


# TODO: if not repeat, choose randomly between 
# not repeating feature or not repeating time point, or not repeating either.

