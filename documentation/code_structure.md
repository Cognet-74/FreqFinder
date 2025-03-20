# Code Structure

This document provides an overview of the FreqFinder codebase structure, explaining the main components and their purposes.

## Package Organization

The FreqFinder package is organized into several modules, each with a specific purpose:

```
freqfinder/
├── __init__.py                  # Package initialization, API functions
├── core/                        # Core functionality
│   ├── __init__.py              # Core module initialization
│   ├── base.py                  # Abstract base classes
│   ├── utils.py                 # Utility functions
│   └── visualization.py         # Visualization functions
├── analysis/                    # Analysis functionality
│   ├── __init__.py              # Analysis module initialization
│   └── stationarity.py          # Stationarity testing functions
├── filters/                     # Filtering functionality
│   ├── __init__.py              # Filters module initialization
│   └── spectral.py              # Filter design and application
├── methods/                     # Implementation of band detection methods
│   ├── __init__.py              # Methods module initialization
│   ├── ufbd.py                  # UFBD implementation
│   ├── inchworm.py              # Inchworm implementation
│   └── hybrid.py                # Hybrid detector implementation
├── original_inchworm.py         # Original implementation of Inchworm algorithm
└── original_ufbd.py             # Original implementation of UFBD algorithm
```

## Main Components

### Top-Level Package (`__init__.py`)

The top-level `__init__.py` file provides the main public API for the package:

- `analyze_bands()`: Main function for analyzing frequency bands
- `plot_analysis()`: Function for visualizing analysis results
- `compare_methods()`: Function for comparing different analysis methods
- `segment_analysis()`: Function for analyzing stationarity in segments

These functions provide a user-friendly interface to the underlying implementation classes.

### Core Module

The `core` module contains the fundamental building blocks of the framework:

#### `core/base.py`

Defines the abstract base classes that form the foundation of the framework:

- `FrequencyBandDetector`: Abstract base class for frequency band detection methods
- `SignalDecomposer`: Abstract base class for signal decomposition into frequency bands
- `FrequencyBandAnalyzer`: Combined interface for frequency band analysis

These abstract classes ensure that all method implementations follow a consistent interface.

#### `core/utils.py`

Contains utility functions for data format conversion and other helper functions:

- Format conversion between different band representations
- Signal processing utilities
- Data validation functions

#### `core/visualization.py`

Contains functions for visualizing frequency bands and analysis results:

- `plot_detected_bands()`: Plot detected frequency bands
- `plot_band_components()`: Plot original time series and band components
- `plot_time_frequency()`: Plot time-frequency representation

### Analysis Module

The `analysis` module contains tools for analyzing time series data:

#### `analysis/stationarity.py`

Contains functions for testing stationarity in time series data:

- `test_stationarity()`: Test time series for stationarity using statistical tests
- `test_band_components_stationarity()`: Test stationarity of each frequency band component
- `segment_stationarity_analysis()`: Analyze stationarity in different segments of the time series
- `plot_stationarity_results()`: Plot stationarity test results for band components
- `plot_segment_stationarity()`: Plot stationarity results for different segments

### Filters Module

The `filters` module contains tools for designing and applying filters:

#### `filters/spectral.py`

Functions for designing and applying filters to extract frequency band components:

- `design_filters()`: Design filters for given frequency bands
- `apply_filter()`: Apply filter to decompose signal into band component

### Methods Module

The `methods` module contains the implementations of different frequency band detection methods:

#### `methods/ufbd.py`

Implementation of the Unsupervised Frequency Band Discovery method:

- `UnsupervisedFrequencyBandDiscovery`: UFBD implementation with unified interface

#### `methods/inchworm.py`

Implementation of the Inchworm Frequency Band Analysis method:

- `InchwormFEBA`: Adapter for the Inchworm FEBA implementation with unified interface

#### `methods/hybrid.py`

Implementation of hybrid approaches combining multiple methods:

- `HybridDetector`: Hybrid approach combining Inchworm's statistical detection with UFBD's feature-based refinement
- `AutoSelectAnalyzer`: Auto-selects the best method based on data characteristics

### Original Algorithm Implementations

The package includes the original implementations of the algorithms:

- `original_inchworm.py`: Original implementation of the Inchworm FEBA algorithm
- `original_ufbd.py`: Original implementation of the UFBD algorithm

These are maintained for reference and are wrapped by the adapter classes in the `methods` module to conform to the unified API.

## File Relationships and Dependencies

The package follows a clear dependency hierarchy:

1. **Core Abstract Classes**: Define the interfaces (`core/base.py`)
2. **Method Implementations**: Implement those interfaces (`methods/*.py`)
3. **Integration Layer**: Brings everything together (`__init__.py`)

Key relationships:

- Method implementers inherit from abstract base classes
- Analysis tools consume method results
- Visualization functions display results from various components
- The top-level API orchestrates interactions between components

## Extension Points

FreqFinder is designed to be extended in several ways:

1. **New Detection Methods**: Create new classes that inherit from `FrequencyBandDetector`
2. **New Decomposition Methods**: Create new classes that inherit from `SignalDecomposer`
3. **New Analysis Tools**: Add new functions to the `analysis` module
4. **New Visualization Functions**: Add new functions to `core/visualization.py`

When extending the framework, follow these best practices:

- Ensure new methods implement all abstract methods from the base classes
- Follow the consistent return format for compatibility with existing visualization tools
- Add appropriate error handling and input validation
- Document the new functionality thoroughly

## Initialization and Configuration

The package uses lazy imports to minimize startup time and memory usage. The main functionality is only imported when needed.

Configuration is primarily done through function parameters, with sensible defaults provided for convenience.

## Future Structure Considerations

The codebase may evolve in the following ways:

1. **Plugin Architecture**: For easier extension with custom methods
2. **Configuration System**: For global settings and defaults
3. **Performance Optimizations**: Cython or Numba acceleration for compute-intensive parts
4. **Multiprocessing Support**: For parallelizing analysis across multiple cores

## Next Steps

To understand how to use these components, see the [API Reference](api_reference.md) and [Usage Examples](usage_examples.md).
