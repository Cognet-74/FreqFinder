# Project Overview

FreqFinder is a comprehensive Python framework for frequency band discovery, signal decomposition, and stationarity testing in time series data. This document provides a high-level overview of the project, its goals, and key components.

## Purpose and Motivation

Time series data often contains oscillatory components at different frequencies that can provide valuable insights. Detecting these frequency bands automatically and decomposing signals into these components are challenging tasks, especially when:

- The data is non-stationary
- The frequency bands are not known in advance
- The bands may overlap or have varying strengths
- Traditional fixed-band approaches may not capture the natural structure of the data

FreqFinder addresses these challenges by providing a unified framework that integrates multiple methodologies for data-driven frequency band discovery.

## Key Features

### Multiple Detection Methods

FreqFinder implements several complementary approaches to frequency band detection:

1. **Unsupervised Frequency Band Discovery (UFBD)**
   - Uses clustering algorithms to group frequencies with similar characteristics
   - Data-driven approach without strong statistical assumptions
   - Effective for detecting bands with clear power distinctions

2. **Inchworm Frequency Band Analysis (FEBA)**
   - Statistical approach that adapts to the data's natural frequency structure
   - Uses hypothesis testing to identify significant frequency boundaries
   - Particularly effective for non-stationary time series

3. **Hybrid Approaches**
   - Combines strengths of multiple methods
   - More robust to different types of data
   - Can leverage statistical testing with feature-based refinement

4. **Auto-Selection**
   - Intelligently selects the best method based on data characteristics
   - Analyzes signal properties to determine the most appropriate approach
   - Saves users from having to choose a method manually

### Signal Decomposition

Once frequency bands are detected, FreqFinder provides tools to:

- Decompose signals into band-specific components
- Design optimal filters for extraction
- Visualize the decomposition results
- Analyze the properties of each component

### Stationarity Testing

FreqFinder integrates stationarity testing capabilities:

- ADF (Augmented Dickey-Fuller) and KPSS (Kwiatkowski-Phillips-Schmidt-Shin) tests
- Testing of both original signals and band components
- Segmented stationarity analysis for evolving signals
- Visualization of stationarity test results

### Advanced Visualization

The framework includes comprehensive visualization tools:

- Frequency band plots with power spectrum overlays
- Time-frequency representations with band boundaries
- Component visualization with stationarity information
- Method comparison visualizations

## Framework Architecture

FreqFinder is built on a modular architecture with several key components:

1. **Abstract Base Classes**: Define common interfaces for all methods
2. **Method Implementations**: Concrete implementations of different detection approaches
3. **Analysis Tools**: Utilities for stationarity testing and other analyses
4. **Filtering Components**: Tools for signal decomposition
5. **Visualization Tools**: Components for visualizing results

The framework uses a layered design:

```
  +-----------------+
  |  Public API     |  High-level functions for easy use
  +-----------------+
          |
  +-----------------+
  | Analyzer Classes|  Implementations of different methods
  +-----------------+
          |
+----------+----------+
|          |          |
v          v          v
+--------+ +--------+ +--------+
|Detector| |Decomp. | |Analysis|  Core components
+--------+ +--------+ +--------+
```

## Applications

FreqFinder is designed for a wide range of applications:

- **Neuroscience**: Analyzing EEG, MEG, or LFP data to identify frequency bands
- **Finance**: Detecting cycles and patterns in market data
- **Climate Science**: Identifying oscillatory patterns in climate signals
- **Engineering**: Analyzing vibration or acoustic data
- **Economics**: Finding cyclical patterns in economic indicators
- **Health Monitoring**: Analyzing physiological signals like heart rate or respiration

## Technical Foundations

FreqFinder is built on solid technical foundations:

- **Spectral Analysis**: Leveraging Fourier transforms and spectral techniques
- **Statistical Testing**: Using robust hypothesis testing for band identification
- **Machine Learning**: Employing clustering and other unsupervised techniques
- **Signal Processing**: Implementing advanced filtering and decomposition methods
- **Object-Oriented Design**: Using a flexible, extensible architecture

## Future Directions

The project is actively evolving with plans for:

- Additional detection methods
- Real-time processing capabilities
- Multivariate time series analysis
- Deep learning integration


## Getting Started

To get started with FreqFinder, proceed to the [Installation Guide](installation_guide.md) and then explore the [Usage Examples](usage_examples.md).
