# freqfinder/analysis/__init__.py

from .stationarity import (
    test_stationarity,
    test_band_components_stationarity,
    segment_stationarity_analysis,
    plot_stationarity_results,
    plot_segment_stationarity,
    use_advanced_stationarity_methods
)

__all__ = [
    'test_stationarity',
    'test_band_components_stationarity',
    'segment_stationarity_analysis',
    'plot_stationarity_results',
    'plot_segment_stationarity',
    'use_advanced_stationarity_methods'
]

# Try to import advanced methods if available
try:
    from .advanced.stationarity import compute_hurst_exponent
    __all__.append('compute_hurst_exponent')
except ImportError:
    pass