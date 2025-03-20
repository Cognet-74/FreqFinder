# File: freqfinder/methods/inchworm.py

from ..core.base import FrequencyBandAnalyzer
from ..core.utils import inchworm_to_standard, standard_to_inchworm
from ..filters.spectral import design_filters, apply_filter
from ..original_inchworm import InchwormFEBA as OrigInchworm
import numpy as np

class InchwormFEBA(FrequencyBandAnalyzer):
    """
    Adapter for the Inchworm FEBA implementation with unified interface.
    """
    
    def __init__(self, alpha=0.05, N=256, K=5, ndraw=1000, block_diag=True, 
                test_stationarity=False, **kwargs):
        """
        Initialize the Inchworm FEBA adapter.
        
        Parameters
        ----------
        alpha : float
            Significance level for hypothesis testing
        N : int
            Window size
        K : int
            Number of tapers
        ndraw : int
            Number of random draws
        block_diag : bool
            Whether to use block diagonal approximation for covariance matrix
        test_stationarity : bool
            Whether to test stationarity of band components
        **kwargs
            Additional parameters for Inchworm FEBA
        """
        super().__init__()
        
        # Store parameters
        self.alpha = alpha
        self.N = N
        self.K = K
        self.ndraw = ndraw
        self.block_diag = block_diag
        self.test_stationarity = test_stationarity
        self.kwargs = kwargs
        
        # Initialize the original Inchworm implementation
        self.original_inchworm = OrigInchworm(
            alpha=alpha,
            block_diag=block_diag
        )
        
        # Store bands and filters
        self.bands = None
        self.filters = None
    
    def detect_bands(self, time_series, sampling_rate=1.0, **kwargs):
        """
        Detect frequency bands using Inchworm FEBA.
        
        Parameters
        ----------
        time_series : np.ndarray
            Input time series data
        sampling_rate : float
            Sampling rate in Hz
        **kwargs
            Additional method-specific parameters
            
        Returns
        -------
        dict
            Dictionary with band information
        """
        # Ensure time_series is properly formatted for the original implementation
        # Original inchworm implementation expects a 2D array with shape (time_points, variables)
        if isinstance(time_series, np.ndarray):
            if time_series.ndim == 1:
                # Convert 1D array to 2D array with one column
                time_series = time_series.reshape(-1, 1)
        else:
            # Convert to numpy array if not already
            time_series = np.array(time_series)
            if time_series.ndim == 1:
                time_series = time_series.reshape(-1, 1)
        
        # Run inchworm search
        results = self.original_inchworm.inchworm_search(
            time_series,
            N=self.N,
            K=self.K,
            ndraw=self.ndraw,
            **kwargs
        )
        
        # Convert to standardized band format
        self.bands = inchworm_to_standard(results)
        
        # Design filters for band components
        self.filters = design_filters(self.bands, sampling_rate)
        
        return self.bands
    
    def decompose(self, time_series, bands=None, sampling_rate=1.0, **kwargs):
        """
        Decompose signal into frequency bands.
        
        Parameters
        ----------
        time_series : np.ndarray
            Input time series data
        bands : dict, optional
            Band information. If None, use previously detected bands
        sampling_rate : float
            Sampling rate in Hz
        **kwargs
            Additional method-specific parameters
            
        Returns
        -------
        dict
            Dictionary with band components
        """
        # Use provided bands if specified, otherwise use previously detected bands
        if bands is not None:
            self.bands = bands
            # Design filters for the provided bands
            self.filters = design_filters(self.bands, sampling_rate)
        
        # Check if bands have been detected
        if self.bands is None or self.filters is None:
            raise ValueError("No bands detected. Run detect_bands first or provide bands.")
        
        # Apply filters to decompose signal
        components = {}
        for band_name, band_filter in self.filters.items():
            components[band_name] = apply_filter(time_series, band_filter)
        
        return components
    
    def get_band_info(self):
        """
        Get information about detected bands.
        
        Returns
        -------
        dict
            Dictionary with band information
        """
        return self.bands