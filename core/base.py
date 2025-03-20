# File: freqfinder/core/base.py

from abc import ABC, abstractmethod
import numpy as np

class FrequencyBandDetector(ABC):
    """Abstract base class for frequency band detection methods."""
    
    @abstractmethod
    def detect_bands(self, time_series, sampling_rate=1.0, **kwargs):
        """
        Detect frequency bands in the given time series.
        
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
            Dictionary containing detected bands information
        """
        pass
    
    @abstractmethod
    def get_band_info(self):
        """
        Get information about detected bands.
        
        Returns
        -------
        dict
            Dictionary with band information
        """
        pass


class SignalDecomposer(ABC):
    """Abstract base class for signal decomposition into frequency bands."""
    
    @abstractmethod
    def decompose(self, time_series, bands, **kwargs):
        """
        Decompose time series into frequency band components.
        
        Parameters
        ----------
        time_series : np.ndarray
            Input time series data
        bands : dict
            Band information (from any detector method)
        **kwargs
            Additional method-specific parameters
            
        Returns
        -------
        dict
            Dictionary with band components
        """
        pass


class FrequencyBandAnalyzer(FrequencyBandDetector, SignalDecomposer):
    """Combined interface for frequency band analysis."""
    
    def analyze(self, time_series, sampling_rate=1.0, test_stationarity=False, **kwargs):
        """
        Perform complete frequency band analysis.
        
        Parameters
        ----------
        time_series : np.ndarray
            Input time series data
        sampling_rate : float
            Sampling rate in Hz
        test_stationarity : bool
            Whether to test stationarity of band components
        **kwargs
            Additional method-specific parameters
            
        Returns
        -------
        dict
            Dictionary with complete analysis results
        """
        # Input validation
        if not isinstance(time_series, (list, np.ndarray)):
            raise ValueError("time_series must be a list or numpy array")
            
        # Ensure time_series is a numpy array
        if isinstance(time_series, list):
            time_series = np.array(time_series)
            
        # Convert to 1D array if needed
        if len(time_series.shape) > 1:
            # If it's a 2D array with a single column or row, flatten it
            if time_series.shape[0] == 1 or time_series.shape[1] == 1:
                time_series = time_series.flatten()
            # Otherwise, keep first dimension if it's a multi-dimensional array
            else:
                time_series = time_series[:, 0] if time_series.shape[1] > 0 else time_series
        
        try:
            # Detect frequency bands
            bands = self.detect_bands(time_series, sampling_rate, **kwargs)
            
            # Ensure bands is a dictionary
            if bands is None:
                bands = {}
            
            # For each band, ensure central_freq is defined
            for band_name, band_info in bands.items():
                if 'central_freq' not in band_info:
                    # Calculate central frequency as the midpoint
                    band_info['central_freq'] = (band_info['min_freq'] + band_info['max_freq']) / 2
                    
            # Decompose signal into band components
            try:
                components = self.decompose(time_series, bands, sampling_rate=sampling_rate, **kwargs)
            except Exception as e:
                # If decomposition fails, provide empty components
                components = {}
            
            results = {
                'bands': bands,
                'components': components
            }
            
            # Perform stationarity testing if requested
            if test_stationarity and components:
                try:
                    from ..analysis.stationarity import test_band_components_stationarity
                    results['stationarity'] = test_band_components_stationarity(components)
                except Exception as e:
                    # If stationarity testing fails, record the error
                    results['stationarity'] = {'error': str(e)}
            
            return results
        except Exception as e:
            # Return a standardized error structure
            return {
                'error': str(e),
                'bands': {},
                'components': {}
            }