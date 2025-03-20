# File: freqfinder/methods/hybrid.py

from ..core.base import FrequencyBandAnalyzer
from ..filters.spectral import design_filters, apply_filter
import numpy as np
from .ufbd import UnsupervisedFrequencyBandDiscovery
from .inchworm import InchwormFEBA
from ..analysis.stationarity import test_band_components_stationarity

class HybridDetector(FrequencyBandAnalyzer):
    """
    Hybrid approach combining Inchworm's statistical detection with
    UFBD's feature-based refinement.
    """
    
    def __init__(self, alpha=0.05, min_bands=2, max_bands=8, 
                N=256, K=5, ndraw=1000, filter_type='auto',
                test_stationarity=False, **kwargs):
        """
        Initialize the hybrid detector.
        
        Parameters
        ----------
        alpha : float
            Significance level for Inchworm detection
        min_bands : int
            Minimum number of bands to consider
        max_bands : int
            Maximum number of bands to consider
        N : int
            Window size for Inchworm
        K : int
            Number of tapers for Inchworm
        ndraw : int
            Number of random draws for Inchworm
        filter_type : str
            Type of filter to use
        test_stationarity : bool
            Whether to test stationarity of band components
        **kwargs
            Additional parameters
        """
        super().__init__()
        
        # Store parameters
        self.alpha = alpha
        self.min_bands = min_bands
        self.max_bands = max_bands
        self.N = N
        self.K = K
        self.ndraw = ndraw
        self.filter_type = filter_type
        self.test_stationarity = test_stationarity
        self.kwargs = kwargs
        
        # Initialize component methods
        self.inchworm = InchwormFEBA(
            alpha=alpha, 
            N=N, 
            K=K, 
            ndraw=ndraw,
            test_stationarity=test_stationarity
        )
        
        self.ufbd = UnsupervisedFrequencyBandDiscovery(
            min_bands=min_bands, 
            max_bands=max_bands,
            filter_type=filter_type,
            test_stationarity=test_stationarity
        )
        
        # Store bands and filters
        self.bands = None
        self.filters = None
    
    def detect_bands(self, time_series, sampling_rate=1.0, **kwargs):
        """
        Two-stage band detection:
        1. Initial detection with Inchworm for statistical significance
        2. Refinement with UFBD for feature-based optimization
        
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
        # Stage 1: Inchworm statistical detection
        inchworm_bands = self.inchworm.detect_bands(time_series, sampling_rate)
        
        # Count number of bands detected by Inchworm
        num_inchworm_bands = len(inchworm_bands)
        
        # Stage 2: Determine approach based on results
        if num_inchworm_bands < self.min_bands:
            # Not enough statistical bands - use UFBD to detect minimum bands
            final_bands = self.ufbd.detect_bands(
                time_series, 
                sampling_rate, 
                min_bands=self.min_bands
            )
        elif num_inchworm_bands > self.max_bands:
            # Too many statistical bands - use UFBD to constrain to maximum
            # Extract frequency boundaries from Inchworm bands
            boundaries = self._extract_boundaries(inchworm_bands)
            
            # Use UFBD with initial boundaries
            final_bands = self.ufbd.detect_bands(
                time_series, 
                sampling_rate,
                max_bands=self.max_bands,
                initial_boundaries=boundaries
            )
        else:
            # Appropriate number of bands - refine with UFBD feature analysis
            final_bands = self._refine_bands(time_series, inchworm_bands, sampling_rate)
        
        # Store final bands
        self.bands = final_bands
        
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
    
    def analyze(self, time_series, sampling_rate=1.0, test_stationarity=False, **kwargs):
        """
        Enhanced analysis that includes stationarity information.
        
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
            Dictionary with complete analysis results and enhanced stationarity information
        """
        # Detect bands if not already done
        if self.bands is None:
            self.detect_bands(time_series, sampling_rate, **kwargs)
        
        # Decompose signal into band components
        components = self.decompose(time_series, sampling_rate=sampling_rate, **kwargs)
        
        # Create results dictionary
        results = {
            'bands': self.bands,
            'components': components
        }
        
        # Perform stationarity testing if requested
        if test_stationarity:
            # Use imported function directly
            results['stationarity'] = test_band_components_stationarity(components)
            
            # Add stationarity information to band info
            enhanced_bands = {}
            
            for band_name, band_info in results['bands'].items():
                enhanced_bands[band_name] = band_info.copy()
                
                # Add stationarity information if available
                if band_name in results['stationarity']:
                    stationarity_info = results['stationarity'][band_name]
                    enhanced_bands[band_name]['is_stationary'] = stationarity_info.get('is_stationary', None)
                    enhanced_bands[band_name]['adf_pvalue'] = stationarity_info.get('adf_pvalue', None)
                    enhanced_bands[band_name]['kpss_pvalue'] = stationarity_info.get('kpss_pvalue', None)
            
            # Replace bands with enhanced version
            results['bands'] = enhanced_bands
            
            # Group bands by stationarity for easier access
            results['stationary_bands'] = {
                name: info for name, info in enhanced_bands.items()
                if info.get('is_stationary', False)
            }
            
            results['nonstationary_bands'] = {
                name: info for name, info in enhanced_bands.items()
                if not info.get('is_stationary', True)
            }
        
        return results
    
    def _extract_boundaries(self, bands):
        """
        Extract frequency boundaries from band dictionary.
        
        Parameters
        ----------
        bands : dict
            Dictionary with band information
        
        Returns
        -------
        list
            Sorted list of unique boundary frequencies
        """
        boundaries = set()
        
        for band_info in bands.values():
            boundaries.add(band_info['min_freq'])
            boundaries.add(band_info['max_freq'])
        
        return sorted(list(boundaries))
    
    def _refine_bands(self, time_series, inchworm_bands, sampling_rate):
        """
        Refine statistically significant bands with UFBD features.
        
        Parameters
        ----------
        time_series : np.ndarray
            Input time series data
        inchworm_bands : dict
            Bands detected by Inchworm
        sampling_rate : float
            Sampling rate in Hz
            
        Returns
        -------
        dict
            Refined bands
        """
        # Extract boundaries from Inchworm
        boundaries = self._extract_boundaries(inchworm_bands)
        
        # Use UFBD to refine within these constraints
        try:
            # This would require extending the UFBD class to support initial_boundaries and refine_only
            refined_bands = self.ufbd.detect_bands(
                time_series, 
                sampling_rate,
                initial_boundaries=boundaries,
                refine_only=True
            )
            return refined_bands
        except:
            # Fall back to using Inchworm bands directly if refinement fails
            return inchworm_bands

class AutoSelectAnalyzer(FrequencyBandAnalyzer):
    """
    Auto-selects the best method based on data characteristics.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the auto-select analyzer.
        
        Parameters
        ----------
        **kwargs
            Additional parameters for detector methods
        """
        super().__init__()
        
        # Initialize component methods
        self.ufbd = UnsupervisedFrequencyBandDiscovery(**kwargs)
        self.inchworm = InchwormFEBA(**kwargs)
        self.hybrid = HybridDetector(**kwargs)
        
        # Store selected method
        self.selected_method = None
    
    def detect_bands(self, time_series, sampling_rate=1.0, **kwargs):
        """
        Automatically select and apply the best method based on data characteristics.
        
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
        # Analyze data characteristics
        data_stats = self._analyze_data(time_series)
        
        # Select method based on characteristics
        if data_stats['length'] < 500:
            # Short time series - use UFBD (more efficient)
            self.selected_method = self.ufbd
            print("Auto-selected UFBD method (short time series)")
        elif data_stats['snr'] > 10:
            # High SNR - use Inchworm (statistical power)
            self.selected_method = self.inchworm
            print("Auto-selected Inchworm method (high SNR)")
        else:
            # Default to hybrid approach
            self.selected_method = self.hybrid
            print("Auto-selected Hybrid method (default)")
        
        # Apply selected method
        return self.selected_method.detect_bands(time_series, sampling_rate, **kwargs)
    
    def decompose(self, time_series, bands=None, sampling_rate=1.0, **kwargs):
        """
        Decompose using the selected or specified method.
        
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
        if self.selected_method is None:
            # Default to hybrid if no method has been selected yet
            self.selected_method = self.hybrid
        
        return self.selected_method.decompose(time_series, bands, sampling_rate, **kwargs)
    
    def get_band_info(self):
        """
        Get information about detected bands from selected method.
        
        Returns
        -------
        dict
            Dictionary with band information
        """
        if self.selected_method is None:
            return None
        
        return self.selected_method.get_band_info()
    
    def _analyze_data(self, time_series):
        """
        Analyze data characteristics to determine best method.
        
        Parameters
        ----------
        time_series : np.ndarray
            Input time series data
            
        Returns
        -------
        dict
            Dictionary with data characteristics
        """
        # Calculate basic statistics
        data_length = len(time_series)
        
        # Estimate signal-to-noise ratio
        from scipy import signal
        f, Pxx = signal.welch(time_series)
        
        # Simple SNR estimation based on power spectrum
        signal_power = np.max(Pxx)
        noise_power = np.median(Pxx)
        snr = signal_power / noise_power if noise_power > 0 else float('inf')
        
        return {
            'length': data_length,
            'snr': snr
        }