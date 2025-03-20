# Enhanced Unsupervised Frequency Band Discovery
# Implementation aligned with full specification document

import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

class UnsupervisedFrequencyBandDiscovery:
    """
    Unsupervised Frequency Band Discovery (UFBD) algorithm
    
    This class implements the UFBD algorithm for discovering optimal
    frequency bands directly from time series data.
    """
    def __init__(self, min_bands=2, max_bands=8, filter_type='auto', clustering_method='kmeans', max_iter=300, **kwargs):
        """
        Initialize the UFBD algorithm
        
        Parameters:
        min_bands (int): Minimum number of frequency bands to consider
        max_bands (int): Maximum number of frequency bands to consider
        filter_type (str): Type of filter to use ('auto', 'spectral', 'fir', or 'iir')
        clustering_method (str): Method for clustering frequencies ('kmeans')
        max_iter (int): Maximum number of iterations for clustering algorithm
        **kwargs: Additional parameters to pass to the clustering algorithm
        """
        self.min_bands = min_bands
        self.max_bands = max_bands
        self.filter_type = filter_type
        self.clustering_method = clustering_method
        self.max_iter = max_iter
        self.clustering_params = kwargs  # Store additional clustering parameters
        self.bands = None
        self.filters = None
        self.band_powers = None
        self.band_boundaries = None
        self.optimization_results = None
        self.actual_filter_types = {}
        
    def fit(self, time_series, sampling_rate=1.0, max_bands=None, **kwargs):
        """
        Discover optimal frequency bands from time series data
        
        Parameters:
        time_series (np.array): Input time series
        sampling_rate (float): Sampling rate of the time series
        max_bands (int, optional): Override for the maximum number of bands
        **kwargs: Additional parameters for customization
        
        Returns:
        self: The fitted object
        """
        # Override max_bands if specified
        if max_bands is not None:
            self.max_bands = max_bands
            
        # Store actual time series length for filter design
        self.time_series_length = len(time_series)
        
        # Compute spectral features
        spectrum = self._compute_spectrum(time_series)
        
        # Find optimal number of bands
        n_bands = self._optimize_band_count(spectrum)
        
        # Cluster frequencies into bands
        self.bands = self._cluster_frequencies(spectrum, n_bands)
        
        # Design filters for each band
        self.filters = self._design_filters(self.bands, sampling_rate)
        
        # Calculate band powers
        self.band_powers = self._calculate_band_powers(spectrum, self.bands)
        
        # Extract band boundaries for compatibility with existing code
        self.band_boundaries = []
        for band_name, band_info in self.bands.items():
            self.band_boundaries.append((band_info['min_freq'], band_info['max_freq']))
        
        # Sort by frequency
        self.band_boundaries.sort()
        
        return self
        
    def transform(self, time_series):
        """
        Decompose time series into discovered frequency bands
        
        Parameters:
        time_series (np.array): Input time series
        
        Returns:
        dict: Dictionary with band components
        """
        if self.filters is None:
            raise ValueError("Model must be fitted before transform")
            
        # Apply each filter to decompose the signal
        decomposed = {}
        for band_name, band_filter in self.filters.items():
            decomposed[band_name] = self._apply_filter(time_series, band_filter)
            
        return decomposed
    
    def fit_transform(self, time_series, sampling_rate=1.0, max_bands=None, **kwargs):
        """
        Discover bands and decompose in one step
        
        Parameters:
        time_series (np.array): Input time series
        sampling_rate (float): Sampling rate of the time series
        max_bands (int, optional): Override for the maximum number of bands
        **kwargs: Additional parameters for customization
        
        Returns:
        dict: Dictionary with band components
        """
        return self.fit(time_series, sampling_rate, max_bands, **kwargs).transform(time_series)
        
    def _compute_spectrum(self, time_series):
        """
        Compute power spectrum and related features
        
        Parameters:
        time_series (np.array): Input time series
        
        Returns:
        dict: Dictionary with spectral features
        """
        # Preprocess time series
        ts = self._preprocess_time_series(time_series)
        
        # Apply window function
        window = signal.windows.hann(len(ts))
        ts_windowed = ts * window
        
        # Compute FFT
        n = len(ts)
        fft_result = np.fft.fft(ts_windowed)
        
        # Keep only positive frequencies (up to Nyquist frequency)
        n_pos = n // 2 + 1
        freqs = np.fft.fftfreq(n)[:n_pos]
        power = np.abs(fft_result[:n_pos])**2
        
        # Log-scale the power to handle wide dynamic range
        log_power = np.log1p(power)
        
        # Normalize frequencies to [0,1]
        norm_freqs = freqs / (freqs[-1] if freqs[-1] != 0 else 1e-10)
        
        # Create feature vectors for clustering
        features = np.column_stack([norm_freqs, log_power])
        
        # Additional features
        # 1. Spectral gradient (rate of change)
        gradient = np.gradient(log_power)
        # 2. Spectral curvature (second derivative)
        curvature = np.gradient(gradient)
        
        # 3. Local spectral entropy (NEW)
        window_size = min(31, n_pos // 3)
        entropy = np.zeros_like(log_power)
        for i in range(n_pos):
            start = max(0, i - window_size // 2)
            end = min(n_pos, i + window_size // 2 + 1)
            window_power = power[start:end]
            total_power = np.sum(window_power)
            if total_power > 0:
                prob = window_power / total_power
                entropy[i] = -np.sum(prob * np.log2(prob + 1e-10))
            else:
                entropy[i] = 0
        
        # Add additional features to feature matrix
        features = np.column_stack([features, gradient, curvature, entropy])
        
        return {
            'frequencies': freqs,
            'power': power,
            'log_power': log_power,
            'features': features,
            'fft_result': fft_result,
            'n_samples': n
        }
    
    def _preprocess_time_series(self, time_series):
        """
        Preprocess time series for spectral analysis
        
        Parameters:
        time_series (np.array): Input time series
        
        Returns:
        np.array: Preprocessed time series
        """
        # Convert to numpy array if needed
        ts = np.asarray(time_series)
        
        # Handle NaN and Inf values
        if np.any(np.isnan(ts)) or np.any(np.isinf(ts)):
            ts = np.nan_to_num(ts, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Remove trend with polynomial detrending
        t = np.arange(len(ts))
        polynomial_coeffs = np.polyfit(t, ts, 1)  # Linear detrending
        trend = np.polyval(polynomial_coeffs, t)
        detrended = ts - trend
        
        # Normalize to zero mean and unit variance
        mean = np.mean(detrended)
        std = np.std(detrended)
        if std > 0:
            normalized = (detrended - mean) / std
        else:
            normalized = detrended - mean
        
        return normalized
        
    def _optimize_band_count(self, spectrum):
        """
        Find optimal number of bands using clustering metrics
        
        Parameters:
        spectrum (dict): Spectral features
        
        Returns:
        int: Optimal number of bands
        """
        features = spectrum['features']
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Evaluate different numbers of clusters
        silhouette_scores = []
        davies_bouldin_scores = []
        inertia_values = []
        
        for k in range(self.min_bands, self.max_bands + 1):
            # Apply k-means clustering with additional parameters
            kmeans_params = {'n_clusters': k, 'random_state': 42, 'max_iter': self.max_iter, 'n_init': 10}
            kmeans_params.update(self.clustering_params)  # Add user-provided parameters
            
            kmeans = KMeans(**kmeans_params)
            cluster_labels = kmeans.fit_predict(scaled_features)
            
            # Calculate metrics for cluster quality
            if len(np.unique(cluster_labels)) > 1:
                silhouette_avg = silhouette_score(scaled_features, cluster_labels)
                silhouette_scores.append(silhouette_avg)
                
                davies_bouldin = davies_bouldin_score(scaled_features, cluster_labels)
                davies_bouldin_scores.append(davies_bouldin)
                
                inertia_values.append(kmeans.inertia_)
            else:
                silhouette_scores.append(-1)
                davies_bouldin_scores.append(float('inf'))
                inertia_values.append(float('inf'))
        
        # Find elbow point in inertia curve (NEW)
        if len(inertia_values) > 2:
            inertia_grad = np.gradient(inertia_values)
            inertia_curv = np.gradient(inertia_grad)
            elbow_idx = np.argmax(inertia_curv) if len(inertia_curv) > 0 else 0
            elbow_k = self.min_bands + elbow_idx
        else:
            elbow_k = self.min_bands
        
        # Find maximum silhouette score
        if len(silhouette_scores) > 0:
            sil_idx = np.argmax(silhouette_scores)
            sil_k = self.min_bands + sil_idx
        else:
            sil_k = self.min_bands
        
        # Find minimum Davies-Bouldin index
        if len(davies_bouldin_scores) > 0:
            db_idx = np.argmin(davies_bouldin_scores)
            db_k = self.min_bands + db_idx
        else:
            db_k = self.min_bands
        
        # Combine metrics with weights (NEW)
        weighted_k = round(0.5 * sil_k + 0.3 * elbow_k + 0.2 * db_k)
        best_k = max(self.min_bands, min(self.max_bands, weighted_k))
        
        # Store optimization results
        self.optimization_results = {
            'silhouette_scores': silhouette_scores,
            'davies_bouldin_scores': davies_bouldin_scores,
            'inertia_values': inertia_values,
            'elbow_k': elbow_k,
            'silhouette_k': sil_k,
            'davies_bouldin_k': db_k,
            'weighted_k': weighted_k,
            'best_k': best_k
        }
        
        return best_k
            
    def _cluster_frequencies(self, spectrum, n_bands):
        """
        Cluster frequency bins into bands
        
        Parameters:
        spectrum (dict): Spectral features
        n_bands (int): Number of bands to create
        
        Returns:
        dict: Dictionary with band information
        """
        features = spectrum['features']
        frequencies = spectrum['frequencies']
        power = spectrum['power']
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Apply k-means clustering with additional parameters
        kmeans_params = {'n_clusters': n_bands, 'random_state': 42, 'max_iter': self.max_iter, 'n_init': 10}
        kmeans_params.update(self.clustering_params)  # Add user-provided parameters
        
        kmeans = KMeans(**kmeans_params)
        cluster_labels = kmeans.fit_predict(scaled_features)
        
        # Process each cluster to define frequency bands
        bands = {}
        for i in range(n_bands):
            # Get indices of frequencies in this cluster
            indices = np.where(cluster_labels == i)[0]
            
            # Skip empty clusters
            if len(indices) == 0:
                continue
            
            # Get frequencies and powers in this cluster
            cluster_freqs = frequencies[indices]
            cluster_powers = power[indices]
            
            # Calculate band characteristics
            min_freq = np.min(cluster_freqs)
            max_freq = np.max(cluster_freqs)
            
            # Calculate weighted central frequency
            if np.sum(cluster_powers) > 0:
                central_freq = np.average(cluster_freqs, weights=cluster_powers)
            else:
                central_freq = np.mean(cluster_freqs)
                
            total_power = np.sum(cluster_powers)
            
            # Check if band is contiguous (allowing small gaps) (NEW)
            sorted_indices = np.argsort(cluster_freqs)
            sorted_freqs = cluster_freqs[sorted_indices]
            freq_diffs = np.diff(sorted_freqs)
            
            if len(freq_diffs) > 0:
                max_gap = np.max(freq_diffs)
                avg_gap = np.mean(freq_diffs)
                # Heuristic for "small gap"
                is_contiguous = max_gap <= 3 * avg_gap
            else:
                is_contiguous = True
            
            # Handle non-contiguous bands by splitting into segments (NEW)
            segments = None
            if not is_contiguous and len(sorted_freqs) > 3:
                # Find gaps larger than threshold
                gap_threshold = 3 * avg_gap
                large_gaps = np.where(freq_diffs > gap_threshold)[0]
                
                # Split into segments at large gaps
                segments = []
                start_idx = 0
                for gap_idx in large_gaps:
                    segments.append({
                        'min_freq': sorted_freqs[start_idx],
                        'max_freq': sorted_freqs[gap_idx]
                    })
                    start_idx = gap_idx + 1
                
                # Add last segment
                segments.append({
                    'min_freq': sorted_freqs[start_idx],
                    'max_freq': sorted_freqs[-1]
                })
            
            # Store band information
            band_name = f'band_{i+1}'
            bands[band_name] = {
                'min_freq': min_freq,
                'max_freq': max_freq,
                'central_freq': central_freq,
                'bandwidth': max_freq - min_freq,
                'indices': indices,
                'total_power': total_power,
                'is_contiguous': is_contiguous,
                'segments': segments
            }
        
        # Sort bands by central frequency (low to high)
        sorted_bands = dict(sorted(
            bands.items(), 
            key=lambda item: item[1]['central_freq']
        ))
        
        # Rename bands to more descriptive names
        band_names = ['ultra_low', 'very_low', 'low', 'medium_low', 
                     'medium', 'medium_high', 'high', 'very_high']
        
        renamed_bands = {}
        for i, (_, band_info) in enumerate(sorted_bands.items()):
            if i < len(band_names):
                new_name = band_names[i]
            else:
                new_name = f'band_{i+1}'
            
            renamed_bands[new_name] = band_info
        
        return renamed_bands
        
    def _design_filters(self, bands, sampling_rate):
        """
        Design filters for each frequency band
        
        Parameters:
        bands (dict): Dictionary with band information
        sampling_rate (float): Sampling rate of the time series
        
        Returns:
        dict: Dictionary with filters for each band
        """
        # Use the actual time series length stored during fit
        time_series_length = getattr(self, 'time_series_length', None)
        
        filters = {}
        
        for band_name, band_info in bands.items():
            # Check if the band is contiguous
            if band_info.get('is_contiguous', True):
                # Get band frequencies
                min_freq = band_info['min_freq']
                max_freq = band_info['max_freq']
                
                # Create single filter for contiguous band
                filters[band_name] = self._design_bandpass_filter(
                    min_freq, max_freq, sampling_rate, time_series_length
                )
                
                # Track the actual filter type used
                self.actual_filter_types[band_name] = filters[band_name]['type']
            else:
                # Create filter bank for non-contiguous band
                segments = band_info.get('segments', None)
                if segments and len(segments) > 0:
                    # Create filter for each segment
                    segment_filters = []
                    for segment in segments:
                        seg_filter = self._design_bandpass_filter(
                            segment['min_freq'],
                            segment['max_freq'],
                            sampling_rate, 
                            time_series_length
                        )
                        segment_filters.append(seg_filter)
                    
                    # Create filter bank
                    filters[band_name] = {
                        'type': 'filter_bank',
                        'segments': segment_filters
                    }
                    
                    # Track the filter type for the bank (based on the first segment type)
                    if segment_filters:
                        self.actual_filter_types[band_name] = f"filter_bank({segment_filters[0]['type']})"
                    else:
                        self.actual_filter_types[band_name] = "filter_bank(empty)"
                else:
                    # Fallback to single filter if segments not available
                    min_freq = band_info['min_freq']
                    max_freq = band_info['max_freq']
                    filters[band_name] = self._design_bandpass_filter(
                        min_freq, max_freq, sampling_rate, time_series_length
                    )
                    
                    # Track the actual filter type used
                    self.actual_filter_types[band_name] = filters[band_name]['type']
            
        return filters
    
    def _design_bandpass_filter(self, low_freq, high_freq, sampling_rate, time_series_length=None):
        """
        Design bandpass filter for given frequency range
        
        Parameters:
        low_freq (float): Lower cutoff frequency
        high_freq (float): Upper cutoff frequency
        sampling_rate (float): Sampling rate of the time series
        time_series_length (int, optional): Length of the time series to filter
        
        Returns:
        dict: Filter parameters
        
        Note: When filter_type is 'auto', this method intelligently selects between
        spectral and FIR/IIR filtering based on time series length, frequency bandwidth,
        and other relevant heuristics.
        """
        # Normalize frequencies to Nyquist
        nyquist = sampling_rate / 2
        low_norm = low_freq / nyquist if nyquist > 0 else 0
        high_norm = high_freq / nyquist if nyquist > 0 else 0.5
        
        # Ensure frequencies are within valid range
        low_norm = max(0.001, min(0.999, low_norm))
        high_norm = max(low_norm + 0.001, min(0.999, high_norm))
        
        # Compute the bandwidth and transition width
        bandwidth = high_norm - low_norm
        transition_width = min(low_norm, 0.999 - high_norm) * 0.5
        transition_width = max(0.001, transition_width)
        
        # Calculate the minimum time series length needed for an FIR filter
        ripple_db = 40.0
        N, _ = signal.kaiserord(ripple_db, transition_width)
        min_length_for_fir = N * 5  # General rule: signal should be at least 5x filter length
        
        # Debug logging to help diagnose filter selection
        debug_info = {
            'time_series_length': time_series_length,
            'min_length_for_fir': min_length_for_fir,
            'bandwidth': bandwidth,
            'transition_width': transition_width,
            'filter_type': self.filter_type
        }
        
        # Determine if we should use spectral filtering
        use_spectral = (
            self.filter_type == 'spectral' or
            (self.filter_type == 'auto' and (
                (time_series_length is not None and time_series_length < min_length_for_fir) or
                transition_width < 0.01 or  # Very narrow transition
                bandwidth < 0.02 or  # Very narrow bandwidth
                N > 1000  # Filter would be excessively long
            ))
        )
        
        if use_spectral:
            # Track the actual filter type used
            filter_result = {
                'type': 'spectral',
                'min_freq': low_freq,
                'max_freq': high_freq,
                'debug_info': debug_info
            }
            return filter_result
        elif self.filter_type == 'fir':
            # Calculate a reasonable filter order
            if time_series_length is not None:
                # Limit filter length to 1/5 of signal length
                N = min(N, time_series_length // 5)
            
            # Additional safety caps
            N = min(N, 501)  # Maximum 501 taps
            N = max(N, 31)   # Minimum 31 taps
            N = 2 * (N // 2) + 1  # Ensure odd filter order
            
            # Get the beta parameter for the Kaiser window
            beta = signal.kaiser_beta(ripple_db)
            
            # Design filter
            taps = signal.firwin(
                N, 
                [low_norm, high_norm], 
                window=('kaiser', beta),
                pass_zero=False,
                scale=True
            )
            
            return {
                'type': 'fir',
                'taps': taps,
                'low_freq': low_freq,
                'high_freq': high_freq,
                'debug_info': debug_info
            }
        elif self.filter_type == 'iir':
            # Use a lower order for IIR
            order = 4
            b, a = signal.butter(order, [low_norm, high_norm], btype='bandpass')
            
            return {
                'type': 'iir',
                'b': b,
                'a': a,
                'low_freq': low_freq,
                'high_freq': high_freq,
                'debug_info': debug_info
            }
        else:
            # For 'auto' or any other unhandled types, default to spectral
            return {
                'type': 'spectral',
                'min_freq': low_freq,
                'max_freq': high_freq,
                'debug_info': debug_info
            }
        
    def _apply_filter(self, time_series, band_filter):
        """
        Apply filter to extract band component
        
        Parameters:
        time_series (np.array): Input time series
        band_filter (dict): Filter parameters
        
        Returns:
        np.array: Filtered time series
        """
        ts = np.asarray(time_series)
        
        # Handle filter banks (NEW)
        if isinstance(band_filter, dict) and band_filter.get('type') == 'filter_bank':
            # Process each segment and combine
            segments = band_filter['segments']
            result = np.zeros_like(ts)
            
            for segment_filter in segments:
                segment_result = self._apply_single_filter(ts, segment_filter)
                result += segment_result
                
            return result
        else:
            # Apply single filter
            return self._apply_single_filter(ts, band_filter)
    
    def _apply_single_filter(self, time_series, band_filter):
        """
        Apply a single filter to the time series
        
        Parameters:
        time_series (np.array): Input time series
        band_filter (dict): Filter parameters
        
        Returns:
        np.array: Filtered time series
        """
        ts = np.asarray(time_series)
        
        filter_type = band_filter['type']
        
        if filter_type == 'spectral':
            # FFT-based filtering
            n = len(ts)
            fft_result = np.fft.fft(ts)
            freqs = np.fft.fftfreq(n)
            
            # Create spectral mask
            min_freq = band_filter['min_freq']
            max_freq = band_filter['max_freq']
            mask = (np.abs(freqs) >= min_freq) & (np.abs(freqs) <= max_freq)
            
            # Apply mask and inverse FFT
            filtered_fft = fft_result.copy()
            filtered_fft[~mask] = 0
            filtered = np.real(np.fft.ifft(filtered_fft))
            
            return filtered
            
        elif filter_type == 'fir':
            # Apply FIR filter
            taps = band_filter['taps']
            filtered = signal.filtfilt(taps, [1.0], ts)
            return filtered
            
        elif filter_type == 'iir':
            # Apply IIR filter
            b = band_filter['b']
            a = band_filter['a']
            filtered = signal.filtfilt(b, a, ts)
            return filtered
            
    def _calculate_band_powers(self, spectrum, bands):
        """
        Calculate power in each frequency band
        
        Parameters:
        spectrum (dict): Spectral features
        bands (dict): Dictionary with band information
        
        Returns:
        dict: Dictionary with power in each band
        """
        power = spectrum['power']
        band_powers = {}
        
        for band_name, band_info in bands.items():
            indices = band_info['indices']
            band_powers[band_name] = np.sum(power[indices])
        
        # Normalize powers to sum to 1
        total_power = sum(band_powers.values())
        if total_power > 0:
            for band_name in band_powers:
                band_powers[band_name] /= total_power
        
        return band_powers
    
    def get_band_info(self):
        """
        Get information about discovered bands
        
        Returns:
        dict: Dictionary with band information
        """
        if self.bands is None:
            return None
        
        band_info = {}
        for band_name, band_data in self.bands.items():
            band_info[band_name] = {
                'min_freq': band_data['min_freq'],
                'max_freq': band_data['max_freq'],
                'central_freq': band_data['central_freq'],
                'bandwidth': band_data['bandwidth'],
                'power': self.band_powers.get(band_name, 0) if self.band_powers else 0,
                'is_contiguous': band_data.get('is_contiguous', True),
                'segments': band_data.get('segments', None),
                'filter_type': self.actual_filter_types.get(band_name, 'unknown')
            }
        
        return band_info
    
    def get_optimization_results(self):
        """
        Get results from the band count optimization process
        
        Returns:
        dict: Dictionary with optimization metrics
        """
        return self.optimization_results
        
    def get_filter_debug_info(self):
        """
        Get debugging information about filter selection decisions
        
        Returns:
        dict: Dictionary with filter selection details
        """
        if self.filters is None:
            return None
            
        debug_info = {}
        for band_name, filter_data in self.filters.items():
            if isinstance(filter_data, dict) and 'debug_info' in filter_data:
                debug_info[band_name] = filter_data['debug_info']
            elif isinstance(filter_data, dict) and filter_data.get('type') == 'filter_bank':
                # For filter banks, get debug info from first segment
                if 'segments' in filter_data and len(filter_data['segments']) > 0:
                    first_segment = filter_data['segments'][0]
                    if 'debug_info' in first_segment:
                        debug_info[band_name] = first_segment['debug_info']
        
        return debug_info