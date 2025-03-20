# Contributing Guidelines

Thank you for your interest in contributing to FreqFinder! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Coding Standards](#coding-standards)
- [Git Workflow](#git-workflow)
- [Pull Request Process](#pull-request-process)
- [Testing Guidelines](#testing-guidelines)
- [Documentation Guidelines](#documentation-guidelines)
- [Issue Reporting](#issue-reporting)
- [Feature Requests](#feature-requests)
- [Community](#community)

## Code of Conduct

All contributors are expected to adhere to the project's Code of Conduct. Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) to ensure a positive and inclusive environment for everyone.

## Getting Started

If you're new to contributing to open source or to FreqFinder specifically, here are some steps to get started:

1. **Familiarize yourself with the project**: Read the [Project Overview](project_overview.md) and explore the [API Reference](api_reference.md) to understand the project's purpose and structure.

2. **Find an issue to work on**: Look for issues labeled `good-first-issue` or `help-wanted` in the [issue tracker](https://github.com/yourusername/freqfinder/issues).

3. **Discuss your approach**: Before making significant changes, discuss your approach in the relevant issue or create a new issue to get feedback.

## Development Environment

### Setting Up Your Environment

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/your-username/freqfinder.git
   cd freqfinder
   ```

2. **Create a virtual environment**:
   ```bash
   # Using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Using conda
   conda create -n freqfinder-env python=3.9
   conda activate freqfinder-env
   ```

3. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

### Development Tools

The project uses several tools to maintain code quality:

- **pytest**: For running tests
- **flake8**: For code style checking
- **black**: For code formatting
- **isort**: For import sorting
- **mypy**: For type checking
- **sphinx**: For documentation generation

You can install all development tools with:

```bash
pip install pytest flake8 black isort mypy sphinx
```

## Coding Standards

### Python Style Guide

FreqFinder follows the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide with some project-specific adaptations:

- **Line length**: Maximum line length is 100 characters.
- **Docstrings**: Use NumPy-style docstrings.
- **Imports**: Use absolute imports and organize them with isort.
- **Quotes**: Use single quotes for short strings and double quotes for docstrings.
- **Type hints**: Use type hints for function signatures when practical.

### Code Formatting

Use black and isort to ensure consistent code formatting:

```bash
# Format code with black
black freqfinder tests

# Sort imports with isort
isort freqfinder tests
```

### Code Quality

Use flake8 and mypy to check code quality:

```bash
# Check with flake8
flake8 freqfinder tests

# Check with mypy
mypy freqfinder
```

## Git Workflow

### Branching Model

FreqFinder uses a simplified Git flow:

- **main**: The main branch contains the stable code that has been released.
- **develop**: The development branch contains the latest development changes.
- **feature/**: Feature branches are used for developing new features.
- **bugfix/**: Bugfix branches are used for fixing bugs.
- **docs/**: Documentation branches are used for updating documentation.

### Branch Naming

Branch names should be descriptive and follow this pattern:

- `feature/short-description`: For new features
- `bugfix/issue-number-short-description`: For bug fixes
- `docs/topic-name`: For documentation changes

### Commit Messages

Write clear, concise commit messages that describe what the commit does:

- Start with a short (50 chars or less) summary in the imperative mood
- Optionally, follow with a blank line and a more detailed explanation
- Reference issues or pull requests as needed

Example:

```
Add UFBD clustering optimization for large datasets

This commit implements an optimized clustering algorithm for the UFBD method
when dealing with large datasets. It reduces memory usage and improves
performance by using mini-batch K-means.

Closes #42
```

## Pull Request Process

1. **Create a new branch** from the `develop` branch for your work.
2. **Make your changes** and commit them with clear messages.
3. **Write or update tests** to cover your changes.
4. **Update documentation** as needed.
5. **Run tests locally** to ensure they pass.
6. **Push your branch** to your fork.
7. **Submit a pull request** to the `develop` branch of the main repository.
8. **Respond to feedback** and make any requested changes.

### Pull Request Template

When submitting a pull request, please use the provided template, which includes:

- A description of the changes
- The motivation and context for the changes
- How to test the changes
- Any breaking changes
- Checklist of completed items

## Testing Guidelines

### Writing Tests

- Write tests for all new code
- Aim for high test coverage (80%+ for new code)
- Test both normal operation and edge cases
- Follow the existing test structure (see [Test Suite Overview](test_suite_overview.md))

### Running Tests

```bash
# Run all tests
pytest

# Run specific tests
pytest tests/unit/test_ufbd.py

# Run tests with coverage
pytest --cov=freqfinder
```

### Continuous Integration

The project uses GitHub Actions for continuous integration. All pull requests must pass the CI checks before they can be merged.

## Documentation Guidelines

### Code Documentation

- Document all public functions, classes, and methods
- Use NumPy-style docstrings
- Include parameter descriptions, return values, and examples
- Update docstrings when changing function behavior

Example:

```python
def analyze_bands(time_series, sampling_rate=1.0, method='auto', test_stationarity=False, **kwargs):
    """
    Analyze frequency bands in time series using specified method.
    
    Parameters
    ----------
    time_series : np.ndarray
        Input time series data
    sampling_rate : float
        Sampling rate in Hz
    method : str or FrequencyBandAnalyzer
        Method to use ('ufbd', 'inchworm', 'hybrid', 'auto')
    test_stationarity : bool
        Whether to test stationarity of band components
    **kwargs
        Additional parameters for the selected method
        
    Returns
    -------
    dict
        Analysis results
        
    Examples
    --------
    >>> import numpy as np
    >>> from freqfinder import analyze_bands
    >>> t = np.arange(0, 10, 0.01)
    >>> signal = np.sin(2 * np.pi * 5 * t)
    >>> results = analyze_bands(signal, sampling_rate=100)
    """
```

### Project Documentation

- Update README.md when adding significant features
- Add or update documentation in the `documentation` directory
- Include examples for new functionality
- Keep API documentation up-to-date

## Issue Reporting

### Bug Reports

When reporting a bug, please include:

- A clear and descriptive title
- A detailed description of the bug
- Steps to reproduce the behavior
- Expected behavior
- Actual behavior
- System information (OS, Python version, etc.)
- Any relevant code, error messages, or screenshots

### Feature Requests

When requesting a feature, please include:

- A clear and descriptive title
- A detailed description of the feature
- The motivation for the feature
- Example use cases
- Potential implementation approaches (if applicable)

## Feature Requests

We welcome feature requests and ideas for improving FreqFinder. When proposing a new feature:

1. **Check existing issues** to see if the feature has already been requested.
2. **Create a new issue** using the feature request template.
3. **Be specific** about what the feature should do and why it's valuable.
4. **Consider scope** - is this a small enhancement or a major new component?

## Community

### Getting Help

If you need help with your contribution:

- Ask questions in the issue you're working on
- Reach out to the maintainers
- Join the project's communication channels (if available)

### Acknowledgment

Contributors will be acknowledged in the project's CONTRIBUTORS.md file. We appreciate all contributions, from code to documentation to bug reports.

## Extension Development

### Adding a New Detection Method

To add a new frequency band detection method:

1. Create a new file in the `freqfinder/methods/` directory.
2. Implement a class that inherits from `FrequencyBandAnalyzer`.
3. Implement the required methods: `detect_bands`, `decompose`, and `get_band_info`.
4. Add appropriate tests in the `tests/unit/` directory.
5. Update the `__init__.py` file to expose the new method.
6. Add documentation for the new method.

Example structure:

```python
# freqfinder/methods/new_method.py

from ..core.base import FrequencyBandAnalyzer

class NewMethodDetector(FrequencyBandAnalyzer):
    """Implementation of a new frequency band detection method."""
    
    def __init__(self, param1=default1, param2=default2, **kwargs):
        """Initialize the detector with custom parameters."""
        self.param1 = param1
        self.param2 = param2
        # Initialize other attributes
        
    def detect_bands(self, time_series, sampling_rate=1.0, **kwargs):
        """Detect frequency bands using the new method."""
        # Implementation
        return bands
        
    def decompose(self, time_series, bands=None, sampling_rate=1.0, **kwargs):
        """Decompose signal into frequency bands."""
        # Implementation
        return components
        
    def get_band_info(self):
        """Get information about detected bands."""
        # Implementation
        return band_info
```

### Adding a New Visualization

To add a new visualization:

1. Add a new function to `freqfinder/core/visualization.py`.
2. Add appropriate tests in `tests/unit/test_visualization.py`.
3. Update the `plot_analysis` function to include the new visualization type.
4. Add documentation for the new visualization.

Example:

```python
# In freqfinder/core/visualization.py

def plot_new_visualization(results, **kwargs):
    """
    Create a new type of visualization for analysis results.
    
    Parameters
    ----------
    results : dict
        Analysis results from analyze_bands
    **kwargs
        Additional visualization parameters
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure with the visualization
    """
    import matplotlib.pyplot as plt
    
    # Implementation
    
    return fig

# Update plot_analysis function to include the new visualization type
def plot_analysis(results, plot_type='bands', **kwargs):
    # ...
    
    elif plot_type == 'new_visualization':
        fig = plot_new_visualization(results, **kwargs)
        return fig
        
    # ...
```

## Conclusion

Thank you for contributing to FreqFinder! Your efforts help improve the project for everyone. If you have questions or need clarification on these guidelines, please reach out to the maintainers.
