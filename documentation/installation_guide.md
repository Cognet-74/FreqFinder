# Installation Guide

This guide covers the installation process for the FreqFinder package, including prerequisites, installation steps, and verification.

## Prerequisites

Before installing FreqFinder, ensure you have the following:

1. **Python**: FreqFinder requires Python 3.7 or higher.
2. **Package Manager**: pip or conda should be installed.
3. **Development Environment**: A code editor or IDE (like VSCode, PyCharm, or Jupyter).

## Dependencies

FreqFinder depends on the following Python packages:

- **numpy**: Numerical computing
- **scipy**: Scientific computing and signal processing
- **matplotlib**: Data visualization
- **statsmodels**: Statistical models and tests
- **scikit-learn**: Machine learning algorithms
- **tqdm**: Progress bar functionality

These dependencies will be automatically installed when you install FreqFinder.

## Installation Methods

### Method 1: Installation from GitHub (Recommended)

For the latest development version:

```bash
# Clone the repository
git clone https://github.com/yourusername/freqfinder.git
cd freqfinder

# Install in development mode
pip install -e .
```

This method is recommended for:
- Users who want to contribute to the project
- Those who need the latest features or bug fixes
- Anyone who wants to customize the code

### Method 2: Installation from PyPI (Coming Soon)

In future releases, FreqFinder will be available on PyPI:

```bash
pip install freqfinder
```

## Virtual Environment (Recommended)

It's a good practice to use a virtual environment for your Python projects. Here's how to set one up:

### Using venv

```bash
# Create a virtual environment
python -m venv freqfinder-env

# Activate the environment (Windows)
freqfinder-env\Scripts\activate

# Activate the environment (macOS/Linux)
source freqfinder-env/bin/activate

# Install FreqFinder
cd path/to/freqfinder
pip install -e .
```

### Using conda

```bash
# Create a conda environment
conda create -n freqfinder-env python=3.9

# Activate the environment
conda activate freqfinder-env

# Install FreqFinder
cd path/to/freqfinder
pip install -e .
```

## Docker Installation (Advanced)

For containerized environments, you can use Docker:

```bash
# Build the Docker image
docker build -t freqfinder .

# Run the container
docker run -it freqfinder
```

## Development Installation

For development purposes, install with additional testing dependencies:

```bash
# Install with development dependencies
pip install -e ".[dev]"
```

## Installation Verification

To verify that FreqFinder is correctly installed, run the following Python code:

```python
import freqfinder
print(freqfinder.__version__)
```

You should see the current version number printed.

Alternatively, you can run the test suite:

```bash
pytest -xvs tests/
```

## Troubleshooting Common Installation Issues

### Issue: Missing Dependencies

If you encounter errors about missing dependencies, try installing them manually:

```bash
pip install numpy scipy matplotlib statsmodels scikit-learn tqdm
```

### Issue: Compilation Errors

If you encounter compilation errors during installation:

1. Ensure you have a C/C++ compiler installed (GCC on Linux/macOS, Microsoft Visual C++ on Windows)
2. Install the required build tools:
   ```bash
   # Windows
   pip install --upgrade setuptools wheel
   
   # Linux/macOS
   pip install --upgrade setuptools wheel
   ```

### Issue: Version Conflicts

If you encounter version conflicts with existing packages:

```bash
# Install in a fresh virtual environment
python -m venv fresh-env
fresh-env\Scripts\activate  # Windows
source fresh-env/bin/activate  # macOS/Linux
pip install -e path/to/freqfinder
```

## Next Steps

After installation, proceed to [Usage Examples](usage_examples.md) to get started with FreqFinder.

For any installation problems not covered here, please open an issue on the [GitHub repository](https://github.com/yourusername/freqfinder/issues).
