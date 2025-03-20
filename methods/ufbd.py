# File: freqfinder/methods/ufbd.py

from ..core.base import FrequencyBandAnalyzer
from ..core.utils import ufbd_to_standard, standard_to_ufbd
from ..filters.spectral import design_filters, apply_filter
import numpy as np

# Import original UFBD and create an alias for testing purposes
from ..original_ufbd import UnsupervisedFrequencyBandDiscovery as OrigUFBD

class UnsupervisedFrequencyBandDiscovery(FrequencyBandAnalyzer):
    """
    Adapter for the UFBD implementation with unified interface.
    """
    
    def __init__(self, min_bands=2, max_bands=8, filter_type='auto', 
                 clustering_method='kmeans', test_stationarity=False, **kwargs):
        """
        Initialize the UFBD adapter.
        
        Parameters
        ----------
        min_bands : int
            Minimum number of bands to consider
        max_bands : int
            Maximum number of bands to consider
        filter_type : str
            Type of filter to use
        clustering_method : str
            Method for clustering frequencies
        test_stationarity : bool
            Whether to test stationarity of band components
        **kwargs
            Additional parameters for UFBD
        """
        super().__init__()
        
        # Import original UFBD implementation
        from ..original_ufbd import UnsupervisedFrequencyBandDiscovery as OrigUFBD
        
        # Store parameters
        self.min_bands = min_bands
        self.max_bands = max_bands
        self.filter_type = filter_type
        self.clustering_method = clustering_method
        self.test_stationarity = test_stationarity
        self.kwargs = kwargs
        
        # Initialize the original UFBD implementation
        self.original_ufbd = OrigUFBD(
            min_bands=min_bands,
            max_bands=max_bands,
            filter_type=filter_type,
            clustering_method=clustering_method,
            **kwargs
        )
        
        # Store bands and filters
        self.bands = None
        self.filters = None
    
    def detect_bands(self, time_series, sampling_rate=1.0, **kwargs):
        """
        Detect frequency bands using UFBD.
        
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
        # Fit the UFBD model to the data
        self.original_ufbd.fit(time_series, sampling_rate=sampling_rate, **kwargs)
        
        # Get band information and convert to standard format
        ufbd_bands = self.original_ufbd.get_band_info()
        
        # Fix negative frequencies and invalid segments before converting to standard format
        if ufbd_bands:
            for band_name, band_info in ufbd_bands.items():
                # Ensure min_freq is non-negative
                if band_info['min_freq'] < 0:
                    band_info['min_freq'] = 0.0
                # Ensure max_freq is less than Nyquist
                if band_info['max_freq'] > sampling_rate/2:
                    band_info['max_freq'] = sampling_rate/2
                # Update central_freq accordingly
                band_info['central_freq'] = (band_info['min_freq'] + band_info['max_freq']) / 2
                # Update bandwidth accordingly
                band_info['bandwidth'] = band_info['max_freq'] - band_info['min_freq']
                
                # Fix segments if present and remove invalid ones
                if band_info.get('segments'):
                    valid_segments = []
                    for segment in band_info['segments']:
                        # Fix segment boundaries
                        if segment['min_freq'] < 0:
                            segment['min_freq'] = 0.0
                        if segment['max_freq'] > sampling_rate/2:
                            segment['max_freq'] = sampling_rate/2
                        # Only keep valid segments
                        if segment['min_freq'] < segment['max_freq']:
                            valid_segments.append(segment)
                    
                    # Replace segments list or set to None if empty
                    if valid_segments:
                        band_info['segments'] = valid_segments
                    else:
                        band_info['segments'] = None
                        band_info['is_contiguous'] = True
        
        # Convert to standardized format
        self.bands = ufbd_to_standard(ufbd_bands)
        
        # Design filters for band components
        self.filters = design_filters(self.bands, sampling_rate, filter_type=self.filter_type)
        
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
            self.filters = design_filters(self.bands, sampling_rate, filter_type=self.filter_type)
        
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