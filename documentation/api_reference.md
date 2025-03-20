# API Reference

This document provides a detailed reference for the FreqFinder API, including functions, classes, and methods.

## Table of Contents

- [Top-Level API](#top-level-api)
- [Core Components](#core-components)
- [Method Implementations](#method-implementations)
- [Analysis Tools](#analysis-tools)
- [Visualization Functions](#visualization-functions)
- [Filtering Components](#filtering-components)

## Top-Level API

The top-level API provides the main interface for using FreqFinder.

### `analyze_bands`

```python
analyze_bands(time_series, sampling_rate=1.0, method='auto', test_stationarity=False, **kwargs)
```

Analyze frequency bands in time series using the specified method.

**Parameters:**
- `time_series` (np.ndarray): Input time series data
- `sampling_rate` (float): Sampling rate in Hz
- `method` (str or FrequencyBandAnalyzer): Method to use ('ufbd', 'inchworm', 'hybrid', 'auto')
- `test_stationarity` (bool): Whether to test stationarity of band components
- `**kwargs`: Additional parameters for the selected method

**Returns:**
- dict: Analysis results containing:
  - `bands`: Dictionary of detected frequency bands
  - `components`: Dictionary of band components
  - `stationarity` (optional): Stationarity test results
  - `method`: Method used for analysis
  - `error` (if error occurred): Error message

**Example:**
```python
import numpy as np
from freqfinder import analyze_bands

# Generate sample data
t = np.arange(0, 10, 0.01)
signal = np.sin(2 * np.pi * 2 * t) + np.sin(2 * np.pi * 5 * t) + 0.1 * np.random.randn(len(t))

# Analyze bands
results = analyze_bands(signal, sampling_rate=100, method='hybrid', test_stationarity=True)

# Print detected bands
for band_name, band_info in results['bands'].items():
    print(f"{band_name}: {band_info['min_freq']:.2f} - {band_info['max_freq']:.2f} Hz")
```

### `plot_analysis`

```python
plot_analysis(results, plot_type='bands', **kwargs)
```

Plot analysis results.

**Parameters:**
- `results` (dict): Results from analyze_bands
- `plot_type` (str): Type of plot ('bands', 'components', 'time_frequency', 'stationarity')
- `**kwargs`: Additional parameters for plotting:
  - For 'bands': `spectrum`, `ax`, `method_name`
  - For 'components': `time_series`, `sampling_rate`, `figsize`
  - For 'time_frequency': `time_series`, `sampling_rate`, `figsize`, `method`
  - For 'stationarity': Parameters for stationarity visualization

**Returns:**
- matplotlib.figure.Figure: Figure with the plot

**Example:**
```python
from freqfinder import analyze_bands, plot_analysis

# Analyze bands
results = analyze_bands(signal, sampling_rate=100, method='hybrid')

# Plot bands
fig1 = plot_analysis(results, plot_type='bands')

# Plot components
fig2 = plot_analysis(results, plot_type='components', 
                     time_series=signal, sampling_rate=100)
```

### `compare_methods`

```python
compare_methods(time_series, sampling_rate=1.0, methods=None, test_stationarity=False, **kwargs)
```

Compare multiple band analysis methods.

**Parameters:**
- `time_series` (np.ndarray): Input time series data
- `sampling_rate` (float): Sampling rate in Hz
- `methods` (list, optional): List of methods to compare (defaults to ['ufbd', 'inchworm', 'hybrid'])
- `test_stationarity` (bool): Whether to test stationarity of band components
- `**kwargs`: Additional parameters for methods

**Returns:**
- dict: Comparison results for each method

**Example:**
```python
from freqfinder import compare_methods

# Compare different methods
comparison = compare_methods(
    signal, 
    sampling_rate=100,
    methods=['ufbd', 'inchworm', 'hybrid'],
    test_stationarity=True
)

# Print number of bands detected by each method
for method, results in comparison.items():
    print(f"{method}: {len(results['bands'])} bands")
```

### `segment_analysis`

```python
segment_analysis(time_series, segments=10, method='both')
```

Analyze stationarity in different segments of the time series.

**Parameters:**
- `time_series` (np.ndarray): Input time series
- `segments` (int): Number of segments to analyze
- `method` (str): Stationarity test method ('adf', 'kpss', or 'both')

**Returns:**
- dict: Segmented stationarity analysis results

**Example:**
```python
from freqfinder import segment_analysis, plot_segment_stationarity

# Analyze stationarity in segments
segment_results = segment_analysis(signal, segments=10, method='both')

# Plot segmented stationarity results
fig = plot_segment_stationarity(segment_results)
```

## Core Components

### Abstract Base Classes

These abstract classes define the interfaces that all method implementations must follow.

#### `FrequencyBandDetector`

```python
class FrequencyBandDetector(ABC)
```

Abstract base class for frequency band detection methods.

**Methods:**
- `detect_bands(time_series, sampling_rate=1.0, **kwargs)`: Detect frequency bands in the given time series.
- `get_band_info()`: Get information about detected bands.

#### `SignalDecomposer`

```python
class SignalDecomposer(ABC)
```

Abstract base class for signal decomposition into frequency bands.

**Methods:**
- `decompose(time_series, bands, **kwargs)`: Decompose time series into frequency band components.

#### `FrequencyBandAnalyzer`

```python
class FrequencyBandAnalyzer(FrequencyBandDetector, SignalDecomposer)
```

Combined interface for frequency band analysis.

**Methods:**
- `analyze(time_series, sampling_rate=1.0, test_stationarity=False, **kwargs)`: Perform complete frequency band analysis.

### Utility Functions

The core module also includes utility functions for data format conversion and other helper functions.

#### `inchworm_to_standard`

```python
inchworm_to_standard(inchworm_results)
```

Convert Inchworm FEBA results to standardized band format.

#### `ufbd_to_standard`

```python
ufbd_to_standard(ufbd_bands)
```

Convert UFBD band info to standardized band format.

#### `standard_to_inchworm`

```python
standard_to_inchworm(standard_bands)
```

Convert standardized bands to Inchworm format.

#### `standard_to_ufbd`

```python
standard_to_ufbd(standard_bands, sampling_rate=1.0)
```

Convert standardized bands to UFBD format.

## Method Implementations

### UFBD Method

#### `UnsupervisedFrequencyBandDiscovery`

```python
class UnsupervisedFrequencyBandDiscovery(FrequencyBandAnalyzer)
```

UFBD implementation with unified interface.

**Parameters:**
- `min_bands` (int): Minimum number of bands to consider
- `max_bands` (int): Maximum number of bands to consider
- `filter_type` (str): Type of filter to use ('auto', 'spectral', 'fir', or 'iir')
- `clustering_method` (str): Method for clustering frequencies
- `test_stationarity` (bool): Whether to test stationarity
- `**kwargs`: Additional parameters

**Methods:**
- `detect_bands(time_series, sampling_rate=1.0, **kwargs)`: Detect frequency bands using UFBD.
- `decompose(time_series, bands=None, sampling_rate=1.0, **kwargs)`: Decompose signal into frequency bands.
- `get_band_info()`: Get information about detected bands.

### Inchworm Method

#### `InchwormFEBA`

```python
class InchwormFEBA(FrequencyBandAnalyzer)
```

Adapter for the Inchworm FEBA implementation with unified interface.

**Parameters:**
- `alpha` (float): Significance level for hypothesis testing
- `N` (int): Window size
- `K` (int): Number of tapers
- `ndraw` (int): Number of random draws
- `block_diag` (bool): Whether to use block diagonal approximation
- `test_stationarity` (bool): Whether to test stationarity
- `**kwargs`: Additional parameters

**Methods:**
- `detect_bands(time_series, sampling_rate=1.0, **kwargs)`: Detect frequency bands using Inchworm FEBA.
- `decompose(time_series, bands=None, sampling_rate=1.0, **kwargs)`: Decompose signal into frequency bands.
- `get_band_info()`: Get information about detected bands.

### Hybrid Methods

#### `HybridDetector`

```python
class HybridDetector(FrequencyBandAnalyzer)
```

Hybrid approach combining Inchworm's statistical detection with UFBD's feature-based refinement.

**Parameters:**
- `alpha` (float): Significance level for hypothesis testing
- `min_bands` (int): Minimum number of bands
- `max_bands` (int): Maximum number of bands
- `N` (int): Window size
- `K` (int): Number of tapers
- `ndraw` (int): Number of random draws
- `filter_type` (str): Type of filter to use
- `test_stationarity` (bool): Whether to test stationarity
- `**kwargs`: Additional parameters

**Methods:**
- `detect_bands(time_series, sampling_rate=1.0, **kwargs)`: Two-stage band detection.
- `decompose(time_series, bands=None, sampling_rate=1.0, **kwargs)`: Decompose signal into frequency bands.
- `get_band_info()`: Get information about detected bands.
- `analyze(time_series, sampling_rate=1.0, test_stationarity=False, **kwargs)`: Enhanced analysis with stationarity information.

#### `AutoSelectAnalyzer`

```python
class AutoSelectAnalyzer(FrequencyBandAnalyzer)
```

Auto-selects the best method based on data characteristics.

**Methods:**
- `detect_bands(time_series, sampling_rate=1.0, **kwargs)`: Automatically select and apply the best method.
- `decompose(time_series, bands=None, sampling_rate=1.0, **kwargs)`: Decompose using the selected method.
- `get_band_info()`: Get information about detected bands from the selected method.

## Analysis Tools

### Stationarity Testing

#### `test_stationarity`

```python
test_stationarity(time_series, method='both', regression='ct')
```

Test time series for stationarity using statistical tests.

**Parameters:**
- `time_series` (np.ndarray): Input time series
- `method` (str): Test method ('adf', 'kpss', or 'both')
- `regression` (str): Regression type for ADF test

**Returns:**
- dict: Stationarity test results

#### `test_band_components_stationarity`

```python
test_band_components_stationarity(components, regression='ct')
```

Test stationarity of each frequency band component.

**Parameters:**
- `components` (dict): Dictionary of band components
- `regression` (str): Regression type for ADF test

**Returns:**
- dict: Stationarity test results for each component

#### `segment_stationarity_analysis`

```python
segment_stationarity_analysis(time_series, segments=10, method='both')
```

Analyze stationarity in different segments of the time series.

**Parameters:**
- `time_series` (np.ndarray): Input time series
- `segments` (int): Number of segments to analyze
- `method` (str): Stationarity test method

**Returns:**
- dict: Segmented stationarity analysis results

## Visualization Functions

### `plot_detected_bands`

```python
plot_detected_bands(bands, spectrum=None, ax=None, method_name=None)
```

Plot detected frequency bands with optional spectrum overlay.

**Parameters:**
- `bands` (dict): Dictionary of detected bands
- `spectrum` (np.ndarray, optional): Power spectrum to overlay
- `ax` (matplotlib.axes.Axes, optional): Axes to plot on
- `method_name` (str, optional): Name of the method used

**Returns:**
- matplotlib.axes.Axes: Axes with the plot

### `plot_band_components`

```python
plot_band_components(time_series, components, stationarity_results=None, sampling_rate=1.0, figsize=(12, 8))
```

Plot original time series and band components with stationarity information.

**Parameters:**
- `time_series` (np.ndarray): Original time series
- `components` (dict): Dictionary of band components
- `stationarity_results` (dict, optional): Stationarity test results
- `sampling_rate` (float): Sampling rate in Hz
- `figsize` (tuple): Figure size

**Returns:**
- matplotlib.figure.Figure: Figure with the plot

### `plot_time_frequency`

```python
plot_time_frequency(time_series, bands=None, sampling_rate=1.0, figsize=(10, 6), method=None)
```

Plot time-frequency representation with optional band overlay.

**Parameters:**
- `time_series` (np.ndarray): Input time series
- `bands` (dict, optional): Dictionary of detected bands
- `sampling_rate` (float): Sampling rate in Hz
- `figsize` (tuple): Figure size
- `method` (str, optional): Method used for visualization

**Returns:**
- matplotlib.figure.Figure: Figure with the plot

### `plot_stationarity_results`

```python
plot_stationarity_results(stationarity_results, figsize=(10, 6))
```

Plot stationarity test results for band components.

**Parameters:**
- `stationarity_results` (dict): Stationarity test results
- `figsize` (tuple): Figure size

**Returns:**
- matplotlib.figure.Figure: Figure with the plot

### `plot_segment_stationarity`

```python
plot_segment_stationarity(segment_results, figsize=(12, 6))
```

Plot stationarity results for different segments.

**Parameters:**
- `segment_results` (dict): Segmented stationarity analysis results
- `figsize` (tuple): Figure size

**Returns:**
- matplotlib.figure.Figure: Figure with the plot

## Filtering Components

### `design_filters`

```python
design_filters(bands, sampling_rate=1.0, filter_type='spectral')
```

Design filters for given frequency bands.

**Parameters:**
- `bands` (dict): Dictionary of frequency bands
- `sampling_rate` (float): Sampling rate in Hz
- `filter_type` (str): Type of filter to use

**Returns:**
- dict: Dictionary of designed filters

### `apply_filter`

```python
apply_filter(time_series, band_filter)
```

Apply filter to decompose signal into band component.

**Parameters:**
- `time_series` (np.ndarray): Input time series
- `band_filter` (dict): Filter parameters

**Returns:**
- np.ndarray: Filtered time series
