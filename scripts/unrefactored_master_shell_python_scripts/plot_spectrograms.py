import matplotlib.pyplot as plt
import scipy.io.wavfile as wf
import numpy as np

base = wf.read("/home/andrew/Documents/PROJECT_FILES/evaluation_materials/final_opt_models_data/16khz_centred/ref_data/hvd_670_absolute_0_CENTRE.wav")


Pxx, freqs, bins, im = plt.specgram(base[1], NFFT=350, Fs=16000)
plt.xscale('linear')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.show()
