# Test Suite Overview

This document provides an overview of the FreqFinder test suite, including the testing framework, test structure, and guidance on running and extending the tests.

## Testing Framework

FreqFinder uses the pytest framework for its test suite. The test suite is organized to ensure comprehensive testing of all components:

- **Unit Tests**: Test individual functions and methods in isolation
- **Integration Tests**: Test interactions between different components
- **Functional Tests**: Test complete workflows from input to output

## Test Directory Structure

The test suite is organized into the following directory structure:

```
tests/
├── __init__.py                      # Test package initialization
├── conftest.py                      # Pytest fixtures and configuration
├── test_analyze_bands.py            # Tests for the analyze_bands function
├── test_compare_methods.py          # Tests for the compare_methods function
├── test_plot_analysis.py            # Tests for the plot_analysis function
├── test_segment_analysis.py         # Tests for the segment_analysis function
├── unit/                            # Unit tests
│   ├── __init__.py
│   ├── test_inchworm.py             # Tests for the Inchworm implementation
│   ├── test_ufbd.py                 # Tests for the UFBD implementation
│   ├── test_hybrid.py               # Tests for the Hybrid implementation
│   ├── test_stationarity.py         # Tests for stationarity analysis
│   └── test_visualization.py        # Tests for visualization functions
├── integration/                     # Integration tests
│   ├── __init__.py
│   ├── test_method_interfaces.py    # Tests for method interfaces
│   └── test_end_to_end.py           # End-to-end tests
└── data/                            # Test data
    ├── __init__.py
    ├── synthetic_data.py            # Functions to generate synthetic test data
    └── sample_data/                 # Sample data files for testing
```

## Test Data

The test suite uses both synthetic and real data:

1. **Synthetic Data**: Generated programmatically with known properties to test specific behaviors:
   - Stationary signals with clear frequency bands
   - Non-stationary signals with time-varying frequency content
   - Signals with noise and artifacts
   - Edge cases with unusual properties

2. **Sample Data**: Small, real-world datasets included in the repository:
   - EEG recordings
   - Financial time series
   - Seismic data
   - Speech signals

## Running Tests

### Basic Test Execution

To run the entire test suite:

```bash
# Run from the project root directory
pytest
```

### Selective Test Execution

To run specific test modules or functions:

```bash
# Run a specific test file
pytest tests/test_analyze_bands.py

# Run a specific test function
pytest tests/test_analyze_bands.py::test_analyze_bands_basic

# Run tests with a specific name pattern
pytest -k "ufbd"

# Run tests in a specific directory
pytest tests/unit/
```

### Test Verbosity and Output

```bash
# Increase verbosity
pytest -v

# Show stdout during tests
pytest -s

# Combine options
pytest -vs

# Generate HTML report
pytest --html=report.html

# Generate code coverage report
pytest --cov=freqfinder
```

## Test Coverage

The test suite aims for comprehensive coverage of all functionality:

- Core functionality: 90%+ coverage target
- Method implementations: 85%+ coverage target
- Visualization functions: 80%+ coverage target

To check test coverage:

```bash
# Install coverage tools
pip install pytest-cov

# Run tests with coverage
pytest --cov=freqfinder --cov-report=term

# Generate HTML coverage report
pytest --cov=freqfinder --cov-report=html
```

## Key Test Fixtures

The test suite uses fixtures to set up test environments and provide data:

### Data Fixtures

```python
@pytest.fixture
def simple_sine_signal():
    """Generate a simple sine wave signal."""
    fs = 100
    t = np.arange(0, 10, 1/fs)
    signal = np.sin(2 * np.pi * 5 * t)
    return signal, fs
    
@pytest.fixture
def complex_test_signal():
    """Generate a complex signal with multiple frequency components."""
    fs = 100
    t = np.arange(0, 10, 1/fs)
    signal = (
        np.sin(2 * np.pi * 2 * t) + 
        0.5 * np.sin(2 * np.pi * 10 * t) + 
        0.2 * np.sin(2 * np.pi * 25 * t)
    )
    return signal, fs
    
@pytest.fixture
def nonstationary_signal():
    """Generate a non-stationary signal with time-varying frequency."""
    fs = 100
    t = np.arange(0, 10, 1/fs)
    chirp = signal.chirp(t, f0=1, f1=20, t1=10, method='linear')
    return chirp, fs
```

### Method Fixtures

```python
@pytest.fixture
def ufbd_detector():
    """Create a UFBD detector with standard parameters."""
    from freqfinder.methods.ufbd import UnsupervisedFrequencyBandDiscovery
    return UnsupervisedFrequencyBandDiscovery(min_bands=2, max_bands=5)
    
@pytest.fixture
def inchworm_detector():
    """Create an Inchworm detector with standard parameters."""
    from freqfinder.methods.inchworm import InchwormFEBA
    return InchwormFEBA(alpha=0.05, N=128, K=5, ndraw=100)
    
@pytest.fixture
def hybrid_detector():
    """Create a Hybrid detector with standard parameters."""
    from freqfinder.methods.hybrid import HybridDetector
    return HybridDetector(alpha=0.05, min_bands=2, max_bands=5)
```

## Test Categories

### Unit Tests

Unit tests focus on testing individual functions and classes in isolation:

- **Core Functions**: Testing utility functions, data conversion, etc.
- **Method Implementations**: Testing the core algorithms of each method
- **Visualization Functions**: Testing plot generation and customization

### Integration Tests

Integration tests focus on the interaction between components:

- **Method Interfaces**: Testing that all methods conform to the common interface
- **API Consistency**: Testing for consistent behavior across the API
- **Error Handling**: Testing error propagation and recovery

### Functional Tests

Functional tests verify end-to-end workflows:

- **Full Analysis Pipeline**: Testing the complete analysis pipeline
- **Method Comparison**: Testing the method comparison functionality
- **Visualization Integration**: Testing the integration of analysis and visualization

## Mocking and Patching

For tests that involve external dependencies or complex components, mocking is used:

```python
@pytest.mark.parametrize("method", ["ufbd", "inchworm", "hybrid"])
def test_analyze_bands_uses_correct_method(method, monkeypatch):
    """Test that analyze_bands uses the correct method."""
    # Create mock method instance
    mock_method = Mock()
    mock_method.analyze.return_value = {'bands': {}, 'components': {}}
    
    # Mock the get_method_instance function
    def mock_get_method_instance(method_name):
        return mock_method
        
    # Apply the patch
    monkeypatch.setattr('freqfinder.get_method_instance', mock_get_method_instance)
    
    # Call the function
    signal = np.sin(np.linspace(0, 10, 1000))
    analyze_bands(signal, method=method)
    
    # Verify the mock was called
    mock_method.analyze.assert_called_once()
```

## Testing Visualization Functions

Testing visualization functions requires special handling:

1. **Non-interactive Backend**: Use a non-interactive Matplotlib backend
2. **Figure Comparison**: Compare figure properties instead of pixel-by-pixel comparison
3. **Output Capture**: Capture and verify the figure output

```python
def test_plot_detected_bands(simple_bands):
    """Test plot_detected_bands function."""
    # Set non-interactive backend
    import matplotlib
    matplotlib.use('Agg')
    
    # Call visualization function
    fig = plot_detected_bands(simple_bands)
    
    # Test that figure was created
    assert isinstance(fig, matplotlib.figure.Figure)
    
    # Test figure properties
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    
    # Test that bands were plotted (there should be vertical lines)
    lines = [l for l in ax.get_lines() if l.get_linestyle() == '--']
    assert len(lines) == len(simple_bands) + 1  # +1 for the 0 Hz line
```

## Error Testing

The test suite includes tests for error conditions and edge cases:

```python
def test_analyze_bands_invalid_input():
    """Test analyze_bands with invalid inputs."""
    # Test with None
    with pytest.raises(TypeError, match="Time series cannot be None"):
        analyze_bands(None)
    
    # Test with empty array
    with pytest.raises(ValueError, match="Time series cannot be empty"):
        analyze_bands([])
    
    # Test with non-numeric data
    with pytest.raises(TypeError):
        analyze_bands(["a", "b", "c"])
    
    # Test with invalid method
    with pytest.raises(ValueError, match="Invalid method"):
        analyze_bands([1, 2, 3], method="invalid_method")
```

## Parameterized Tests

Parameterized tests are used to test multiple scenarios with similar test logic:

```python
@pytest.mark.parametrize(
    "signal_generator,expected_band_count", 
    [
        (lambda: np.sin(2 * np.pi * 5 * np.linspace(0, 10, 1000)), 1),
        (lambda: np.sin(2 * np.pi * 5 * np.linspace(0, 10, 1000)) + 
                np.sin(2 * np.pi * 15 * np.linspace(0, 10, 1000)), 2),
        (lambda: np.sin(2 * np.pi * 5 * np.linspace(0, 10, 1000)) + 
                np.sin(2 * np.pi * 15 * np.linspace(0, 10, 1000)) +
                np.sin(2 * np.pi * 30 * np.linspace(0, 10, 1000)), 3),
    ]
)
def test_band_count_detection(signal_generator, expected_band_count):
    """Test that the correct number of bands is detected."""
    signal = signal_generator()
    results = analyze_bands(signal, sampling_rate=100, method='ufbd')
    assert len(results['bands']) == expected_band_count
```

## Performance Tests

For computationally intensive methods, performance tests ensure reasonable performance:

```python
@pytest.mark.slow
def test_inchworm_performance():
    """Test Inchworm method performance."""
    from freqfinder.methods.inchworm import InchwormFEBA
    import time
    
    # Generate a long signal
    fs = 100
    t = np.arange(0, 60, 1/fs)  # 1 minute of data
    signal = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 15 * t)
    
    # Initialize the detector
    detector = InchwormFEBA(alpha=0.05, N=256, K=5, ndraw=500)
    
    # Measure execution time
    start_time = time.time()
    results = detector.analyze(signal, sampling_rate=fs)
    end_time = time.time()
    
    # Check that execution time is reasonable
    execution_time = end_time - start_time
    assert execution_time < 60  # Should take less than 60 seconds
```

## Continuous Integration

The test suite is designed to run in continuous integration (CI) environments:

- **GitHub Actions**: Configured to run tests on push and pull requests
- **Test Environments**: Tests run on multiple Python versions (3.7, 3.8, 3.9, 3.10)
- **OS Compatibility**: Tests run on Windows, macOS, and Linux

## Adding New Tests

When adding new functionality to FreqFinder, follow these steps to add tests:

1. **Add Unit Tests**: Create tests for individual functions and methods
2. **Add Integration Tests**: Test the interaction with other components
3. **Update Functional Tests**: Ensure end-to-end workflows still work
4. **Test Error Conditions**: Add tests for error handling
5. **Test Edge Cases**: Consider and test edge cases

## Example Test File

Here's an example test file for the `analyze_bands` function:

```python
# tests/test_analyze_bands.py

import numpy as np
import pytest
from unittest.mock import Mock

from freqfinder import analyze_bands

@pytest.fixture
def sine_signal():
    """Generate a sine wave signal."""
    fs = 100
    t = np.arange(0, 10, 1/fs)
    signal = np.sin(2 * np.pi * 5 * t)
    return signal, fs

def test_analyze_bands_basic(sine_signal):
    """Test basic functionality of analyze_bands."""
    signal, fs = sine_signal
    
    # Test with default parameters
    results = analyze_bands(signal, sampling_rate=fs)
    
    # Check that results have the expected structure
    assert isinstance(results, dict)
    assert 'bands' in results
    assert 'components' in results
    assert 'method' in results
    
    # Check that at least one band was detected
    assert len(results['bands']) > 0
    
    # Check that components were created
    assert len(results['components']) == len(results['bands'])
    
    # Test with stationarity testing
    results = analyze_bands(signal, sampling_rate=fs, test_stationarity=True)
    assert 'stationarity' in results

def test_analyze_bands_methods(sine_signal):
    """Test analyze_bands with different methods."""
    signal, fs = sine_signal
    
    for method in ['ufbd', 'inchworm', 'hybrid', 'auto']:
        results = analyze_bands(signal, sampling_rate=fs, method=method)
        assert results['method'] == method
        assert len(results['bands']) > 0

def test_analyze_bands_error_handling():
    """Test error handling in analyze_bands."""
    # Test with invalid signal
    with pytest.raises(TypeError):
        analyze_bands(None)
        
    # Test with invalid sampling rate
    with pytest.raises(ValueError):
        analyze_bands([1, 2, 3], sampling_rate=-1)
        
    # Test with invalid method
    with pytest.raises(ValueError):
        analyze_bands([1, 2, 3], method='invalid_method')

@pytest.mark.parametrize('invalid_input', [
    None,
    [],
    "string",
    np.array([]),
    np.array([np.nan, np.nan]),
    np.array([np.inf, np.inf])
])
def test_analyze_bands_invalid_inputs(invalid_input):
    """Test analyze_bands with various invalid inputs."""
    with pytest.raises((TypeError, ValueError)):
        analyze_bands(invalid_input)
```

## Troubleshooting Tests

Common test failures and their solutions:

1. **Missing Dependencies**: Ensure all testing dependencies are installed
2. **Environment Issues**: Check for environment-specific issues (OS differences, Python versions)
3. **Random Seed Dependencies**: Fix tests that depend on random values by setting seeds
4. **Timeouts**: Optimize performance-intensive tests or mark them as slow
5. **Matplotlib Backend Issues**: Use `matplotlib.use('Agg')` for visualization tests

## Next Steps

To get started with running and extending the test suite:

1. Install the testing dependencies: `pip install -e ".[dev]"`
2. Run the test suite: `pytest`
3. Check test coverage: `pytest --cov=freqfinder`
4. Explore test files to understand the testing approach
5. Add new tests for any new features or bug fixes
