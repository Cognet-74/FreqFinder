# Usage Examples

This document provides practical examples of how to use FreqFinder for various scenarios. Each example includes complete code and explanations.

## Table of Contents

- [Basic Band Detection](#basic-band-detection)
- [Signal Decomposition](#signal-decomposition)
- [Stationarity Testing](#stationarity-testing)
- [Method Comparison](#method-comparison)
- [Customizing Detection Parameters](#customizing-detection-parameters)
- [Working with Real Data](#working-with-real-data)
- [Advanced Visualization](#advanced-visualization)

## Basic Band Detection

This example demonstrates the simplest use case: detecting frequency bands in a synthetic signal.

```python
import numpy as np
import matplotlib.pyplot as plt
from freqfinder import analyze_bands, plot_analysis

# Create a synthetic signal with two frequency bands
sampling_rate = 100  # Hz
t = np.arange(0, 10, 1/sampling_rate)
signal = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 15 * t) + 0.1 * np.random.randn(len(t))

# Analyze bands using the auto-select method
results = analyze_bands(
    signal, 
    sampling_rate=sampling_rate, 
    method='auto'
)

# Print detected bands
print("Detected frequency bands:")
for band_name, band_info in results['bands'].items():
    print(f"{band_name}: {band_info['min_freq']:.2f} - {band_info['max_freq']:.2f} Hz")

# Plot results
fig = plot_analysis(results, plot_type='bands')
plt.show()
```

## Signal Decomposition

This example shows how to decompose a signal into its frequency band components.

```python
import numpy as np
import matplotlib.pyplot as plt
from freqfinder import analyze_bands, plot_analysis

# Create a synthetic signal with three frequency bands
sampling_rate = 200  # Hz
t = np.arange(0, 20, 1/sampling_rate)

# Low frequency component (2 Hz)
signal_low = 1.5 * np.sin(2 * np.pi * 2 * t)

# Medium frequency component (10 Hz)
signal_med = np.sin(2 * np.pi * 10 * t)

# High frequency component (30 Hz)
signal_high = 0.5 * np.sin(2 * np.pi * 30 * t)

# Combine components with noise
signal = signal_low + signal_med + signal_high + 0.1 * np.random.randn(len(t))

# Analyze bands and decompose signal
results = analyze_bands(
    signal, 
    sampling_rate=sampling_rate, 
    method='hybrid'
)

# Plot original signal and band components
fig = plot_analysis(
    results, 
    plot_type='components', 
    time_series=signal, 
    sampling_rate=sampling_rate
)
plt.show()

# Access individual components
components = results['components']
print(f"Number of components: {len(components)}")

# Plot spectrum of each component
plt.figure(figsize=(12, 8))
for i, (band_name, component) in enumerate(components.items(), 1):
    plt.subplot(len(components), 1, i)
    
    # Compute power spectrum
    from scipy import signal as sig
    f, Pxx = sig.welch(component, fs=sampling_rate, nperseg=1024)
    
    plt.semilogy(f, Pxx)
    plt.title(f"Component: {band_name}")
    plt.ylabel("Power")
    
plt.xlabel("Frequency (Hz)")
plt.tight_layout()
plt.show()
```

## Stationarity Testing

This example demonstrates how to test the stationarity of a signal and its frequency band components.

```python
import numpy as np
import matplotlib.pyplot as plt
from freqfinder import analyze_bands, plot_analysis, segment_analysis, plot_segment_stationarity

# Create a non-stationary signal
sampling_rate = 100  # Hz
t = np.arange(0, 30, 1/sampling_rate)

# Increasing amplitude in low frequency
signal_low = t/30 * np.sin(2 * np.pi * 2 * t)

# Decreasing amplitude in medium frequency
signal_med = (1 - t/30) * np.sin(2 * np.pi * 10 * t)

# Combined non-stationary signal
signal = signal_low + signal_med + 0.1 * np.random.randn(len(t))

# Analyze bands with stationarity testing
results = analyze_bands(
    signal, 
    sampling_rate=sampling_rate, 
    method='hybrid',
    test_stationarity=True  # Enable stationarity testing
)

# Print stationarity test results
print("Stationarity test results:")
if 'stationarity' in results:
    for band_name, stat_results in results['stationarity'].items():
        adf_stationary = stat_results.get('adf', {}).get('stationary', False)
        kpss_stationary = stat_results.get('kpss', {}).get('stationary', False)
        
        print(f"{band_name}:")
        print(f"  ADF: {'Stationary' if adf_stationary else 'Non-stationary'}")
        print(f"  KPSS: {'Stationary' if kpss_stationary else 'Non-stationary'}")

# Plot stationarity results
fig = plot_analysis(results, plot_type='stationarity')
plt.show()

# Perform segmented stationarity analysis
segment_results = segment_analysis(signal, segments=10, method='both')

# Plot segmented stationarity results
fig = plot_segment_stationarity(segment_results)
plt.show()
```

## Method Comparison

This example shows how to compare different frequency band detection methods.

```python
import numpy as np
import matplotlib.pyplot as plt
from freqfinder import compare_methods, plot_analysis

# Create a signal with multiple frequency bands
sampling_rate = 100  # Hz
t = np.arange(0, 20, 1/sampling_rate)

# Base components
signal_low = np.sin(2 * np.pi * 1 * t)
signal_med = 0.8 * np.sin(2 * np.pi * 8 * t)
signal_high = 0.5 * np.sin(2 * np.pi * 20 * t)

# Add a transient event in the middle
transient = np.zeros_like(t)
transient_idx = (t > 8) & (t < 12)
transient[transient_idx] = 0.7 * np.sin(2 * np.pi * 35 * t[transient_idx])

# Combine components with noise
signal = signal_low + signal_med + signal_high + transient + 0.1 * np.random.randn(len(t))

# Compare different methods
comparison = compare_methods(
    signal, 
    sampling_rate=sampling_rate,
    methods=['ufbd', 'inchworm', 'hybrid'],
    test_stationarity=True
)

# Print comparison results
print("Method comparison:")
for method, results in comparison.items():
    print(f"\n{method.upper()}:")
    print(f"  Number of bands: {len(results['bands'])}")
    
    # Print band information
    for band_name, band_info in results['bands'].items():
        print(f"  {band_name}: {band_info['min_freq']:.2f} - {band_info['max_freq']:.2f} Hz")

# Plot bands for each method
plt.figure(figsize=(15, 10))
for i, (method, results) in enumerate(comparison.items(), 1):
    plt.subplot(len(comparison), 1, i)
    ax = plot_analysis(results, plot_type='bands', method_name=method.upper())
    
plt.tight_layout()
plt.show()

# Plot time-frequency representation for the best method (hybrid)
hybrid_results = comparison['hybrid']
fig = plot_analysis(
    hybrid_results, 
    plot_type='time_frequency',
    time_series=signal,
    sampling_rate=sampling_rate,
    method='hybrid'
)
plt.show()
```

## Customizing Detection Parameters

This example shows how to customize the parameters of the detection methods.

```python
import numpy as np
import matplotlib.pyplot as plt
from freqfinder import analyze_bands, plot_analysis

# Create a test signal
sampling_rate = 250  # Hz
t = np.arange(0, 10, 1/sampling_rate)

# Complex signal with multiple components
signal = (
    1.0 * np.sin(2 * np.pi * 2 * t) +
    0.8 * np.sin(2 * np.pi * 8 * t) +
    0.6 * np.sin(2 * np.pi * 15 * t) +
    0.4 * np.sin(2 * np.pi * 30 * t) +
    0.2 * np.sin(2 * np.pi * 60 * t) +
    0.1 * np.random.randn(len(t))
)

# Customize UFBD parameters
ufbd_results = analyze_bands(
    signal, 
    sampling_rate=sampling_rate, 
    method='ufbd',
    min_bands=4,        # Minimum number of bands to consider
    max_bands=8,        # Maximum number of bands to consider
    filter_type='fir',  # Use FIR filters
    clustering_method='kmeans',
    max_iter=500        # Maximum iterations for clustering
)

print("UFBD bands with custom parameters:")
for band_name, band_info in ufbd_results['bands'].items():
    print(f"{band_name}: {band_info['min_freq']:.2f} - {band_info['max_freq']:.2f} Hz")

# Customize Inchworm parameters
inchworm_results = analyze_bands(
    signal, 
    sampling_rate=sampling_rate, 
    method='inchworm',
    alpha=0.05,        # Significance level for testing
    N=512,             # Window size
    K=7,               # Number of tapers
    ndraw=2000,        # Number of random draws
    block_diag=True    # Use block diagonal approximation
)

print("\nInchworm bands with custom parameters:")
for band_name, band_info in inchworm_results['bands'].items():
    print(f"{band_name}: {band_info['min_freq']:.2f} - {band_info['max_freq']:.2f} Hz")

# Customize Hybrid parameters
hybrid_results = analyze_bands(
    signal, 
    sampling_rate=sampling_rate, 
    method='hybrid',
    alpha=0.1,          # More permissive significance level
    min_bands=3,        # Minimum number of bands
    max_bands=10,       # Maximum number of bands
    N=512,              # Window size
    K=5,                # Number of tapers
    filter_type='auto'  # Automatic filter selection
)

print("\nHybrid bands with custom parameters:")
for band_name, band_info in hybrid_results['bands'].items():
    print(f"{band_name}: {band_info['min_freq']:.2f} - {band_info['max_freq']:.2f} Hz")

# Plot comparison of customized methods
plt.figure(figsize=(15, 12))

plt.subplot(3, 1, 1)
plot_analysis(ufbd_results, plot_type='bands', method_name='UFBD (Custom)')

plt.subplot(3, 1, 2)
plot_analysis(inchworm_results, plot_type='bands', method_name='Inchworm (Custom)')

plt.subplot(3, 1, 3)
plot_analysis(hybrid_results, plot_type='bands', method_name='Hybrid (Custom)')

plt.tight_layout()
plt.show()
```

## Working with Real Data

This example demonstrates how to analyze real-world EEG data.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from freqfinder import analyze_bands, plot_analysis

# Load sample EEG data (replace with your data loading method)
# Example assuming data is in a .mat file:
# data = io.loadmat('eeg_sample.mat')
# eeg_signal = data['eeg_data']
# sampling_rate = data['sampling_rate'][0][0]

# For demonstration, we'll simulate EEG-like data
def simulate_eeg(duration=10, sampling_rate=250):
    """Simulate EEG-like data with realistic frequency components."""
    t = np.arange(0, duration, 1/sampling_rate)
    n_samples = len(t)
    
    # Delta (1-4 Hz)
    delta = 15 * np.sin(2 * np.pi * 2 * t)
    
    # Theta (4-8 Hz)
    theta = 10 * np.sin(2 * np.pi * 6 * t)
    
    # Alpha (8-13 Hz)
    alpha_freq = 10
    alpha_amp = 20
    # Alpha typically appears when eyes are closed, simulate this
    alpha_envelope = np.zeros_like(t)
    alpha_envelope[(t > 3) & (t < 7)] = 1  # Alpha appears between 3-7 seconds
    alpha = alpha_amp * alpha_envelope * np.sin(2 * np.pi * alpha_freq * t)
    
    # Beta (13-30 Hz)
    beta = 5 * np.sin(2 * np.pi * 20 * t)
    
    # Gamma (30+ Hz)
    gamma = 2 * np.sin(2 * np.pi * 40 * t)
    
    # Combine components with noise
    eeg = delta + theta + alpha + beta + gamma + 3 * np.random.randn(n_samples)
    
    return eeg, sampling_rate

# Simulate EEG data
eeg_signal, sampling_rate = simulate_eeg(duration=10, sampling_rate=250)

# Plot the EEG signal
plt.figure(figsize=(12, 4))
t = np.arange(0, len(eeg_signal)) / sampling_rate
plt.plot(t, eeg_signal)
plt.title("EEG Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (μV)")
plt.grid(True)
plt.show()

# Analyze frequency bands in the EEG
eeg_results = analyze_bands(
    eeg_signal, 
    sampling_rate=sampling_rate, 
    method='hybrid',
    test_stationarity=True
)

# Print detected bands
print("Detected EEG frequency bands:")
for band_name, band_info in eeg_results['bands'].items():
    print(f"{band_name}: {band_info['min_freq']:.2f} - {band_info['max_freq']:.2f} Hz")

# Plot band components
fig = plot_analysis(
    eeg_results, 
    plot_type='components', 
    time_series=eeg_signal, 
    sampling_rate=sampling_rate
)
plt.show()

# Plot time-frequency representation
fig = plot_analysis(
    eeg_results, 
    plot_type='time_frequency',
    time_series=eeg_signal,
    sampling_rate=sampling_rate
)
plt.show()

# Map detected bands to standard EEG bands
standard_eeg_bands = {
    'delta': (1, 4),   # 1-4 Hz
    'theta': (4, 8),   # 4-8 Hz
    'alpha': (8, 13),  # 8-13 Hz
    'beta': (13, 30),  # 13-30 Hz
    'gamma': (30, 100) # 30+ Hz
}

# Compare detected bands with standard bands
print("\nComparison with standard EEG bands:")
for std_name, (std_min, std_max) in standard_eeg_bands.items():
    # Find detected bands that overlap with this standard band
    overlapping_bands = []
    for band_name, band_info in eeg_results['bands'].items():
        # Check for overlap
        band_min = band_info['min_freq']
        band_max = band_info['max_freq']
        
        if (band_min <= std_max and band_max >= std_min):
            overlapping_bands.append(band_name)
    
    if overlapping_bands:
        print(f"Standard {std_name} ({std_min}-{std_max} Hz) overlaps with: {', '.join(overlapping_bands)}")
    else:
        print(f"Standard {std_name} ({std_min}-{std_max} Hz) has no corresponding detected band")
```

## Advanced Visualization

This example demonstrates advanced visualization techniques using FreqFinder.

```python
import numpy as np
import matplotlib.pyplot as plt
from freqfinder import analyze_bands, plot_analysis
from scipy import signal

# Create a non-stationary signal with time-varying frequency content
sampling_rate = 200  # Hz
t = np.arange(0, 15, 1/sampling_rate)

# Create frequency-modulated components
chirp_slow = signal.chirp(t, f0=1, f1=10, t1=15, method='linear')
chirp_fast = signal.chirp(t, f0=15, f1=40, t1=15, method='quadratic')

# Create amplitude-modulated component
am_freq = 25  # Hz
am_carrier = np.sin(2 * np.pi * am_freq * t)
am_mod = 0.5 * (1 + np.sin(2 * np.pi * 0.2 * t))  # 0.2 Hz modulation
am_signal = am_mod * am_carrier

# Combine components with noise
complex_signal = 1.5 * chirp_slow + 0.8 * chirp_fast + am_signal + 0.1 * np.random.randn(len(t))

# Analyze bands
results = analyze_bands(
    complex_signal, 
    sampling_rate=sampling_rate, 
    method='hybrid',
    test_stationarity=True
)

# 1. Create custom multi-panel visualization
plt.figure(figsize=(15, 12))

# Panel 1: Original signal
plt.subplot(3, 1, 1)
plt.plot(t, complex_signal)
plt.title("Complex Non-Stationary Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

# Panel 2: Spectrogram
plt.subplot(3, 1, 2)
f, t_spec, Sxx = signal.spectrogram(
    complex_signal, 
    fs=sampling_rate,
    nperseg=256,
    noverlap=200,
    nfft=1024,
    detrend='constant',
    scaling='spectrum'
)
plt.pcolormesh(t_spec, f, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
plt.title("Spectrogram")
plt.ylabel("Frequency (Hz)")
plt.xlabel("Time (s)")
plt.colorbar(label='Power/Frequency (dB/Hz)')

# Panel 3: Detected frequency bands with power spectrum
plt.subplot(3, 1, 3)
plot_analysis(results, plot_type='bands')

plt.tight_layout()
plt.show()

# 2. Customized component visualization with stationarity information
fig = plot_analysis(
    results, 
    plot_type='components', 
    time_series=complex_signal, 
    sampling_rate=sampling_rate,
    figsize=(12, 10)
)

# 3. Create a custom band-filtered spectrogram for the first band
# Get the first band's frequencies
first_band_name = list(results['bands'].keys())[0]
first_band = results['bands'][first_band_name]
min_freq = first_band['min_freq']
max_freq = first_band['max_freq']

plt.figure(figsize=(15, 8))

# Original spectrogram
plt.subplot(2, 1, 1)
plt.pcolormesh(t_spec, f, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
plt.title(f"Full Spectrogram")
plt.ylabel("Frequency (Hz)")
plt.colorbar(label='Power (dB)')

# Band-limited spectrogram
plt.subplot(2, 1, 2)
# Mask frequencies outside the band
band_mask = (f >= min_freq) & (f <= max_freq)
Sxx_band = Sxx.copy()
Sxx_band[~band_mask] = 1e-10  # Set to very small value instead of zero for log scale

plt.pcolormesh(t_spec, f, 10 * np.log10(Sxx_band), shading='gouraud', cmap='plasma')
plt.title(f"Band-Limited Spectrogram: {first_band_name} ({min_freq:.1f}-{max_freq:.1f} Hz)")
plt.ylabel("Frequency (Hz)")
plt.xlabel("Time (s)")
plt.colorbar(label='Power (dB)')

plt.tight_layout()
plt.show()

# 4. Plot band power evolution over time
plt.figure(figsize=(12, 6))

# Get components
components = results['components']

# Calculate band power evolution using rolling window
window_size = int(sampling_rate)  # 1-second window
step_size = int(sampling_rate / 10)  # 0.1-second step

time_points = []
band_powers = {band_name: [] for band_name in components.keys()}

for i in range(0, len(complex_signal) - window_size, step_size):
    segment = complex_signal[i:i+window_size]
    time_points.append((i + window_size/2) / sampling_rate)  # Center of window
    
    # Calculate power in each band for this segment
    for band_name, component in components.items():
        band_segment = component[i:i+window_size]
        # Calculate power (mean squared amplitude)
        power = np.mean(band_segment ** 2)
        band_powers[band_name].append(power)

# Plot band power evolution
for band_name, powers in band_powers.items():
    band_info = results['bands'][band_name]
    label = f"{band_name} ({band_info['min_freq']:.1f}-{band_info['max_freq']:.1f} Hz)"
    plt.plot(time_points, powers, label=label)

plt.title("Band Power Evolution Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Band Power")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
```

These examples demonstrate the flexibility and power of the FreqFinder framework for analyzing frequency content in time series data. For more detailed information about the API, see the [API Reference](api_reference.md).
