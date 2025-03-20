![My Logo](logo.png)

# FreqFinder

![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

FreqFinder is a comprehensive and novel Python framework for frequency band discovery, signal decomposition, and stationarity testing in time series data and integrates statistical and machine learning methods.

## Features

- **Multiple Detection Methods**: Choose from UFBD (clustering-based), Inchworm (statistical), and Hybrid approaches
- **Unified API**: Common interface for all frequency band analysis methods
- **Stationarity Testing**: Integrated ADF and KPSS tests with visual representation
- **Signal Decomposition**: Decompose time series into frequency band components
- **Advanced Visualization**: Comprehensive tools for visualizing results
- **Auto-Selection**: Intelligent method selection based on data characteristics

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/freqfinder.git
cd freqfinder

# Install in development mode
pip install -e .
```

### Dependencies

- numpy
- scipy
- matplotlib
- statsmodels
- scikit-learn
- tqdm

## Quick Start

```python
import numpy as np
from freqfinder import analyze_bands, plot_analysis

# Generate sample data
sampling_rate = 100  # Hz
t = np.arange(0, 10, 1/sampling_rate)
signal = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 15 * t) + 0.1 * np.random.randn(len(t))

# Analyze frequency bands
results = analyze_bands(
    signal, 
    sampling_rate=sampling_rate, 
    method='hybrid',
    test_stationarity=True
)

# Print detected bands
for band_name, band_info in results['bands'].items():
    print(f"{band_name}: {band_info['min_freq']:.2f} - {band_info['max_freq']:.2f} Hz")

# Plot results
plot_analysis(results, plot_type='bands')
plot_analysis(results, plot_type='components', time_series=signal, sampling_rate=sampling_rate)
```

## Method Comparison

You can easily compare different methods to find the one that works best for your data:

```python
from freqfinder import compare_methods

# Compare different methods
comparison = compare_methods(
    signal, 
    sampling_rate=sampling_rate,
    methods=['ufbd', 'inchworm', 'hybrid'],
    test_stationarity=True
)

# Print number of bands detected by each method
for method, results in comparison.items():
    print(f"{method}: {len(results['bands'])} bands")
```

## Available Methods

- **UFBD**: Unsupervised Frequency Band Discovery using clustering algorithms
- **Inchworm**: Statistical approach based on adaptive frequency band detection
- **Hybrid**: Combined approach leveraging strengths of both methods
- **Auto**: Automatic selection of the best method based on data characteristics

## Documentation

For full documentation, see the [documentation directory](documentation/README.md) or visit our [GitHub Pages](https://yourusername.github.io/freqfinder/).

## Examples

The repository includes several examples demonstrating different use cases:

- Basic band detection
- Signal decomposition
- Stationarity testing
- Method comparison
- Customizing detection parameters

## Testing

Run the test suite to ensure everything is working correctly:

```bash
# Run tests
pytest tests/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For issues, questions, or contributions, please open an issue on the [GitHub repository](https://github.com/yourusername/freqfinder/issues).
