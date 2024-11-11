import IPython.display as ipd
import librosa
import noisereduce as nr
import matplotlib.pyplot as plt
import numpy as np
from noisereduce.generate_noise import band_limited_noise

# Load the audio file
file_path = '/content/drive/MyDrive/AmpersandProfiles/sample.wav'
data, sr = librosa.load(file_path, sr=None)

# Display the original audio
ipd.display(ipd.Audio(file_path, autoplay=False))

# Plot the original waveform
plt.figure(figsize=(20, 3))
plt.plot(data, color="blue")
plt.title("Original Audio Waveform")
plt.show()

# Generate band-limited noise between 3000Hz and 18000Hz
noise = band_limited_noise(min_freq=3000, max_freq=18000, samples=len(data), samplerate=sr) * 10

# Add the noise to the original audio
audio_with_noise = data + noise

# Reduce noise from the noisy audio
reduced_noise = nr.reduce_noise(y=audio_with_noise, sr=sr, stationary=False)

# Plot the noisy audio waveform
plt.figure(figsize=(20, 3))
plt.plot(audio_with_noise, color="orange")
plt.title("Audio with Band-Limited Noise")
plt.show()

# Plot the noise-reduced waveform
plt.figure(figsize=(20, 3))
plt.plot(reduced_noise, color="green")
plt.title("Noise-Reduced Audio Waveform")
plt.show()

# Display the noise-reduced audio
ipd.display(ipd.Audio(data=reduced_noise, rate=sr))
