# Troubleshooting Guide

This guide helps you diagnose and resolve common issues encountered when using FreqFinder.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Runtime Errors](#runtime-errors)
- [Method-Specific Issues](#method-specific-issues)
- [Performance Problems](#performance-problems)
- [Visualization Issues](#visualization-issues)
- [Testing Issues](#testing-issues)
- [Debugging Techniques](#debugging-techniques)
- [Reporting Issues](#reporting-issues)

## Installation Issues

### Missing Dependencies

**Problem**: Error messages about missing dependencies during installation or when importing FreqFinder.

**Solution**:
1. Install required dependencies manually:
   ```bash
   pip install numpy scipy matplotlib statsmodels scikit-learn tqdm
   ```
2. Try installing FreqFinder with all dependencies:
   ```bash
   pip install -e .
   ```
3. Check for version conflicts:
   ```bash
   pip list
   ```

### Import Errors

**Problem**: `ImportError` or `ModuleNotFoundError` when trying to import FreqFinder modules.

**Solution**:
1. Ensure FreqFinder is installed:
   ```bash
   pip list | grep freqfinder
   ```
2. Check your Python path:
   ```python
   import sys
   print(sys.path)
   ```
3. If using a development setup, ensure you've installed it in development mode:
   ```bash
   pip install -e .
   ```

### Compilation Errors

**Problem**: C extension compilation errors during installation.

**Solution**:
1. Install required build tools:
   ```bash
   # Windows
   pip install --upgrade setuptools wheel

   # Linux/macOS
   pip install --upgrade setuptools wheel
   ```
2. Ensure you have a C compiler installed:
   - Windows: Install Microsoft Visual C++ Build Tools
   - macOS: Install Xcode Command Line Tools
   - Linux: Install gcc/g++

## Runtime Errors

### Input Validation Errors

**Problem**: Errors about invalid inputs when calling FreqFinder functions.

**Solution**:
1. Check input time series:
   - Ensure it's a 1D numpy array or list
   - Ensure it contains numeric values (no NaN, Inf)
   - Ensure it's not empty
2. Check sampling rate:
   - Must be a positive number
3. Check method specification:
   - Must be one of 'ufbd', 'inchworm', 'hybrid', 'auto', or a custom analyzer instance

### Memory Errors

**Problem**: `MemoryError` or system running out of memory when processing large time series.

**Solution**:
1. Process smaller segments of data
2. Use less memory-intensive methods (e.g., 'ufbd' instead of 'inchworm')
3. Adjust method parameters to use less memory:
   ```python
   results = analyze_bands(
       signal,
       method='inchworm',
       N=128,  # Smaller window size
       ndraw=500  # Fewer random draws
   )
   ```
4. Close other applications to free memory

### Numerical Errors

**Problem**: Numerical issues such as `RuntimeWarning` about divide by zero, invalid value, or overflow.

**Solution**:
1. Preprocess your data to handle extremes:
   ```python
   # Remove NaN/Inf values
   signal = np.nan_to_num(signal)
   
   # Normalize data
   signal = (signal - np.mean(signal)) / np.std(signal)
   ```
2. Add small epsilon values to avoid divide-by-zero:
   ```python
   epsilon = 1e-10
   result = x / (y + epsilon)
   ```

## Method-Specific Issues

### UFBD Issues

**Problem**: UFBD method not detecting expected bands or clustering poorly.

**Solution**:
1. Adjust min_bands and max_bands parameters:
   ```python
   results = analyze_bands(
       signal,
       method='ufbd',
       min_bands=3,  # Minimum number of bands to find
       max_bands=8   # Maximum number of bands to find
   )
   ```
2. Try different clustering methods or parameters:
   ```python
   results = analyze_bands(
       signal,
       method='ufbd',
       clustering_method='kmeans',
       max_iter=500,  # More iterations for better convergence
       filter_type='fir'  # Try different filter types
   )
   ```
3. Preprocess your signal to enhance frequency separation:
   ```python
   from scipy import signal as sig
   
   # Apply a pre-emphasis filter
   preemphasized = sig.lfilter([1, -0.95], [1], raw_signal)
   
   # Then analyze
   results = analyze_bands(preemphasized, method='ufbd')
   ```

### Inchworm Issues

**Problem**: Inchworm method taking too long, using too much memory, or producing too few/many bands.

**Solution**:
1. Adjust alpha (significance level) to control band detection sensitivity:
   ```python
   results = analyze_bands(
       signal,
       method='inchworm',
       alpha=0.1,  # More permissive (0.05 is default)
       block_diag=True  # Use faster approximation
   )
   ```
2. Reduce computational complexity:
   ```python
   results = analyze_bands(
       signal,
       method='inchworm',
       N=256,      # Window size
       K=5,        # Number of tapers (smaller = faster)
       ndraw=500   # Number of random draws (smaller = faster)
   )
   ```
3. For signals with weak bands, increase significance level:
   ```python
   results = analyze_bands(
       signal,
       method='inchworm',
       alpha=0.2  # More permissive for weak signals
   )
   ```

### Hybrid Method Issues

**Problem**: Hybrid method producing inconsistent or unexpected results.

**Solution**:
1. Adjust the balance between methods:
   ```python
   results = analyze_bands(
       signal,
       method='hybrid',
       alpha=0.1,     # Inchworm component parameter
       min_bands=3,   # UFBD component parameter
       max_bands=8    # UFBD component parameter
   )
   ```
2. If hybrid is too slow, consider using individual methods:
   ```python
   # Try each method separately
   ufbd_results = analyze_bands(signal, method='ufbd')
   inchworm_results = analyze_bands(signal, method='inchworm')
   
   # Compare results
   print("UFBD bands:", len(ufbd_results['bands']))
   print("Inchworm bands:", len(inchworm_results['bands']))
   ```

## Performance Problems

### Slow Execution

**Problem**: Analysis taking too long to complete.

**Solution**:
1. Reduce signal length or downsample:
   ```python
   # Downsample signal
   downsampled_signal = signal[::2]  # Take every second sample
   downsampled_fs = original_fs / 2
   
   # Run analysis on downsampled signal
   results = analyze_bands(downsampled_signal, sampling_rate=downsampled_fs)
   ```
2. Choose faster methods:
   - UFBD is generally faster than Inchworm
   - Avoid 'hybrid' method for large datasets
3. Adjust method parameters for speed:
   ```python
   # For UFBD
   results = analyze_bands(
       signal,
       method='ufbd',
       max_iter=100  # Fewer iterations
   )
   
   # For Inchworm
   results = analyze_bands(
       signal,
       method='inchworm',
       ndraw=200,  # Fewer random draws
       N=128       # Smaller window size
   )
   ```

### Memory Usage

**Problem**: High memory usage causing system slowdown or crashes.

**Solution**:
1. Process data in smaller chunks:
   ```python
   # Process in chunks of 10,000 samples
   chunk_size = 10000
   results_list = []
   
   for i in range(0, len(signal), chunk_size):
       chunk = signal[i:i+chunk_size]
       result = analyze_bands(chunk, sampling_rate=fs)
       results_list.append(result)
   
   # Aggregate results (implementation depends on your needs)
   ```
2. Use memory-efficient settings:
   ```python
   results = analyze_bands(
       signal,
       method='ufbd',  # More memory-efficient than inchworm
       filter_type='spectral'  # Most memory-efficient filter type
   )
   ```
3. Free memory when possible:
   ```python
   import gc
   
   # After processing large chunks
   del large_variable
   gc.collect()
   ```

## Visualization Issues

### Plot Display Problems

**Problem**: Visualization functions returning errors or displaying incorrectly.

**Solution**:
1. Check results dictionary structure:
   ```python
   # Debug the results structure
   print(results.keys())
   if 'bands' in results:
       print("Number of bands:", len(results['bands']))
   if 'components' in results:
       print("Number of components:", len(results['components']))
   ```
2. Ensure matplotlib is properly configured:
   ```python
   import matplotlib.pyplot as plt
   
   # Set backend if needed
   import matplotlib
   matplotlib.use('TkAgg')  # Or another backend
   
   # Check configuration
   print(matplotlib.get_backend())
   ```
3. Fix empty or NaN values in plot data:
   ```python
   # Check for NaN or Inf values
   import numpy as np
   print("NaN values in signal:", np.any(np.isnan(signal)))
   print("Inf values in signal:", np.any(np.isinf(signal)))
   
   # Clean data before visualization
   clean_signal = np.nan_to_num(signal)
   ```

### Customization Issues

**Problem**: Difficulty customizing plots or modifying visualization output.

**Solution**:
1. Access and modify the returned figure:
   ```python
   # Get figure from plot_analysis
   fig = plot_analysis(results, plot_type='bands')
   
   # Customize the figure
   ax = fig.axes[0]
   ax.set_title('Custom Title')
   ax.set_xlabel('Custom X Label')
   ax.set_ylabel('Custom Y Label')
   ax.grid(True, linestyle='--', alpha=0.7)
   
   # Change colors
   for line in ax.lines:
       if line.get_linestyle() == '--':
           line.set_color('red')
   
   plt.tight_layout()
   plt.show()
   ```
2. Create custom visualizations using the components:
   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   
   # Get components
   components = results['components']
   
   # Create custom visualization
   fig, axes = plt.subplots(len(components) + 1, 1, figsize=(12, 8), sharex=True)
   
   # Plot original signal
   axes[0].plot(signal)
   axes[0].set_title('Original Signal')
   
   # Plot each component
   for i, (band_name, component) in enumerate(components.items(), 1):
       axes[i].plot(component)
       band_info = results['bands'][band_name]
       title = f"{band_name}: {band_info['min_freq']:.2f}-{band_info['max_freq']:.2f} Hz"
       axes[i].set_title(title)
   
   plt.tight_layout()
   plt.show()
   ```

## Testing Issues

### Test Failures

**Problem**: Test failures when running the test suite.

**Solution**:
1. Update to the latest version of dependencies:
   ```bash
   pip install --upgrade numpy scipy matplotlib statsmodels scikit-learn pytest
   ```
2. Check test environment:
   ```bash
   # Show Python version
   python --version
   
   # Show package versions
   pip list
   ```
3. Run specific test files or tests:
   ```bash
   # Run a specific test file
   pytest tests/test_analyze_bands.py -v
   
   # Run a specific test
   pytest tests/test_analyze_bands.py::test_analyze_bands_basic -v
   ```
4. Debug test failures with more information:
   ```bash
   pytest -xvs tests/test_analyze_bands.py
   ```

### Coverage Issues

**Problem**: Low test coverage or coverage report issues.

**Solution**:
1. Install coverage tools:
   ```bash
   pip install pytest-cov
   ```
2. Run with coverage:
   ```bash
   pytest --cov=freqfinder
   ```
3. Generate detailed coverage report:
   ```bash
   pytest --cov=freqfinder --cov-report=html
   ```

## Debugging Techniques

### Debugging FreqFinder Functions

**Problem**: Need to debug function execution to find issues.

**Solution**:
1. Enable verbose output with print statements:
   ```python
   import numpy as np
   from freqfinder import analyze_bands
   
   # Create test signal
   t = np.linspace(0, 10, 1000)
   signal = np.sin(2 * np.pi * 5 * t)
   
   # Insert debug print statements
   print("Signal shape:", signal.shape)
   print("Signal mean:", np.mean(signal))
   print("Signal std:", np.std(signal))
   
   # Run analysis
   results = analyze_bands(signal, sampling_rate=100)
   
   # Print intermediate results
   print("Method used:", results['method'])
   print("Number of bands:", len(results['bands']))
   for band_name, band_info in results['bands'].items():
       print(f"{band_name}: {band_info['min_freq']:.2f}-{band_info['max_freq']:.2f} Hz")
   ```
2. Use Python's built-in debugger:
   ```python
   import pdb
   
   # Set breakpoint
   pdb.set_trace()
   
   # Run function
   results = analyze_bands(signal, sampling_rate=100)
   
   # At the breakpoint, use:
   # n - next line
   # s - step into function
   # c - continue execution
   # p variable - print variable
   # q - quit debugging
   ```
3. Use logging for less intrusive debugging:
   ```python
   import logging
   
   # Configure logging
   logging.basicConfig(level=logging.DEBUG, 
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
   logger = logging.getLogger('freqfinder_debug')
   
   # Add log messages
   logger.debug("Processing signal of length %d", len(signal))
   
   # Run analysis
   results = analyze_bands(signal, sampling_rate=100)
   
   logger.debug("Analysis completed, found %d bands", len(results['bands']))
   ```

### Diagnosing Algorithm Issues

**Problem**: Need to understand why algorithms are producing unexpected results.

**Solution**:
1. Visualize intermediate results:
   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from scipy import signal as sig
   
   # Compute power spectrum
   f, Pxx = sig.welch(signal, fs=100, nperseg=512)
   
   # Plot power spectrum
   plt.figure(figsize=(10, 6))
   plt.semilogy(f, Pxx)
   plt.title('Power Spectrum')
   plt.xlabel('Frequency (Hz)')
   plt.ylabel('Power/Frequency (dB/Hz)')
   plt.grid(True)
   plt.show()
   
   # Compare with detected bands
   plot_analysis(results, plot_type='bands')
   ```
2. Use synthetic signals with known properties:
   ```python
   import numpy as np
   from freqfinder import analyze_bands
   
   # Create synthetic signal with known bands
   fs = 100
   t = np.arange(0, 10, 1/fs)
   
   # 5 Hz component
   signal_5hz = np.sin(2 * np.pi * 5 * t)
   
   # 15 Hz component
   signal_15hz = 0.5 * np.sin(2 * np.pi * 15 * t)
   
   # Combined signal
   signal = signal_5hz + signal_15hz
   
   # Analyze
   results = analyze_bands(signal, sampling_rate=fs)
   
   # Verify bands match expected frequencies
   for band_name, band_info in results['bands'].items():
       print(f"{band_name}: {band_info['min_freq']:.2f}-{band_info['max_freq']:.2f} Hz")
   ```
3. Incremental testing with simplified cases:
   ```python
   # Test with single frequency
   signal1 = np.sin(2 * np.pi * 5 * t)
   results1 = analyze_bands(signal1, sampling_rate=fs)
   
   # Test with two frequencies
   signal2 = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 15 * t)
   results2 = analyze_bands(signal2, sampling_rate=fs)
   
   # Test with three frequencies
   signal3 = (np.sin(2 * np.pi * 5 * t) + 
              np.sin(2 * np.pi * 15 * t) + 
              np.sin(2 * np.pi * 30 * t))
   results3 = analyze_bands(signal3, sampling_rate=fs)
   
   # Compare results
   print("Bands in signal1:", len(results1['bands']))
   print("Bands in signal2:", len(results2['bands']))
   print("Bands in signal3:", len(results3['bands']))
   ```

## Reporting Issues

If you encounter a persistent issue that you cannot resolve:

1. Check if the issue has already been reported in the [issue tracker](https://github.com/yourusername/freqfinder/issues).
2. Prepare a minimum reproducible example:
   ```python
   # minimum_example.py
   import numpy as np
   from freqfinder import analyze_bands
   
   # Create minimal example that reproduces the issue
   np.random.seed(42)  # For reproducibility
   
   # Generate data
   t = np.linspace(0, 10, 1000)
   signal = np.sin(2 * np.pi * 5 * t)
   
   # Reproduce the issue
   results = analyze_bands(signal, sampling_rate=100, method='ufbd')
   ```
3. Include system information:
   ```python
   import sys
   import numpy
   import scipy
   import matplotlib
   import sklearn
   import freqfinder
   
   print("Python version:", sys.version)
   print("NumPy version:", numpy.__version__)
   print("SciPy version:", scipy.__version__)
   print("Matplotlib version:", matplotlib.__version__)
   print("scikit-learn version:", sklearn.__version__)
   print("FreqFinder version:", freqfinder.__version__)
   ```
4. Submit the issue with:
   - Clear description of the problem
   - Minimum reproducible example
   - Expected behavior
   - Actual behavior
   - System information
   - Error messages and stack traces
   - Any relevant plots or visualizations

By following this troubleshooting guide, you should be able to diagnose and resolve most issues with FreqFinder. If you find new issues or have suggestions for improving the troubleshooting process, please contribute to the documentation.
