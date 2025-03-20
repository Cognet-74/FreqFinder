# File: freqfinder/filters/spectral.py

import numpy as np
from scipy import signal

def design_filters(bands, sampling_rate=1.0, filter_type='spectral'):
    """
    Design filters for given frequency bands.
    
    Parameters
    ----------
    bands : dict
        Band information in standardized format
    sampling_rate : float
        Sampling rate in Hz
    filter_type : str
        Type of filter to use ('spectral', 'fir', 'iir', or 'auto')
        'auto' will select an appropriate filter type based on the data
        
    Returns
    -------
    dict
        Dictionary with filters for each band
    """
    filters = {}
    
    # Handle 'auto' filter type selection
    resolved_filter_type = filter_type
    if filter_type == 'auto':
        # Default to spectral filtering for auto mode
        # In auto mode, we choose based on band characteristics
        # For simplicity, we'll use spectral filtering as the default
        resolved_filter_type = 'spectral'
        
        # Optional logic to choose filter type based on band properties
        # If most bands are narrow (high Q factor), IIR might be better
        # If bands are wide with steep transitions, FIR might be better
        narrow_bands = 0
        for _, band_info in bands.items():
            band_width = band_info['max_freq'] - band_info['min_freq']
            center_freq = (band_info['max_freq'] + band_info['min_freq']) / 2
            if band_width < 0.1 * center_freq:  # Narrow band criterion
                narrow_bands += 1
                
        # If most bands are narrow, use IIR
        if narrow_bands > len(bands) / 2:
            resolved_filter_type = 'iir'
    
    # Input validation for all bands
    for band_name, band_info in bands.items():
        min_freq = band_info['min_freq']
        max_freq = band_info['max_freq']
        
        # Catch invalid frequency ranges
        if min_freq < 0:
            raise ValueError(f"Band '{band_name}' has negative min_freq: {min_freq}")
        if min_freq >= max_freq:
            raise ValueError(f"Band '{band_name}' has min_freq ({min_freq}) >= max_freq ({max_freq})")
        
        # Handle frequencies exceeding Nyquist frequency
        nyquist = sampling_rate/2
        if max_freq > nyquist:
            # Cap the frequency to Nyquist frequency
            original_max = max_freq
            max_freq = nyquist * 0.99  # Slight buffer to avoid exact Nyquist
            import warnings
            warnings.warn(f"Band '{band_name}' has max_freq ({original_max}) > Nyquist frequency ({nyquist}). Capping to {max_freq}")
            band_info['max_freq'] = max_freq
            band_info['capped'] = True  # Flag to indicate this band was modified
            band_info['original_max_freq'] = original_max  # Store original value
            
            # Ensure min_freq is also adjusted if needed to maintain valid range
            if min_freq >= max_freq:
                original_min = min_freq
                min_freq = max(0, max_freq * 0.5)  # Set to half of max_freq or 0
                band_info['min_freq'] = min_freq
                warnings.warn(f"Adjusting min_freq for band '{band_name}' from {original_min} to {min_freq} to maintain valid range")
                band_info['original_min_freq'] = original_min  # Store original value
                
            # Update central_freq and bandwidth
            band_info['central_freq'] = (min_freq + max_freq) / 2
            band_info['bandwidth'] = max_freq - min_freq
        
        # Ensure the central_freq is included for each band
        # This addresses the KeyError: 'central_freq' issue
        if 'central_freq' not in band_info:
            band_info['central_freq'] = (min_freq + max_freq) / 2
        
        # Design filter based on resolved type
        if resolved_filter_type == 'spectral':
            filters[band_name] = {
                'type': 'spectral',
                'min_freq': min_freq,
                'max_freq': max_freq,
                'central_freq': band_info['central_freq']
            }
        elif resolved_filter_type == 'fir':
            # Normalize frequencies to Nyquist
            nyquist = sampling_rate / 2
            low_norm = min_freq / nyquist
            high_norm = max_freq / nyquist
            
            # Ensure frequencies are within valid range for filter design
            low_norm = max(0.001, min(0.999, low_norm))
            high_norm = max(low_norm + 0.001, min(0.999, high_norm))
            
            # Calculate filter parameters
            width = 0.05
            ripple_db = 40.0
            N, beta = signal.kaiserord(ripple_db, width)
            N = 2 * (N // 2) + 1  # Make sure N is odd
            
            # Design FIR filter
            taps = signal.firwin(
                N, 
                [low_norm, high_norm], 
                window=('kaiser', beta),
                pass_zero=False
            )
            
            filters[band_name] = {
                'type': 'fir',
                'taps': taps,
                'central_freq': band_info['central_freq']
            }
        elif resolved_filter_type == 'iir':
            # Normalize frequencies to Nyquist
            nyquist = sampling_rate / 2
            low_norm = min_freq / nyquist
            high_norm = max_freq / nyquist
            
            # Ensure frequencies are within valid range for filter design
            low_norm = max(0.001, min(0.999, low_norm))
            high_norm = max(low_norm + 0.001, min(0.999, high_norm))
            
            # Design IIR filter
            order = 4
            b, a = signal.butter(order, [low_norm, high_norm], btype='bandpass')
            
            filters[band_name] = {
                'type': 'iir',
                'b': b,
                'a': a,
                'central_freq': band_info['central_freq']
            }
        else:
            raise ValueError(f"Unknown filter type: {resolved_filter_type}")
    
    return filters


def apply_filter(time_series, band_filter):
    """
    Apply filter to decompose signal into band component.
    
    Parameters
    ----------
    time_series : np.ndarray
        Input time series data
    band_filter : dict
        Filter parameters
        
    Returns
    -------
    np.ndarray
        Filtered time series
    """
    filter_type = band_filter['type']
    
    if filter_type == 'spectral':
        # FFT-based filtering
        min_freq = band_filter['min_freq']
        max_freq = band_filter['max_freq']
        
        # Compute FFT
        n = len(time_series)
        fft_result = np.fft.fft(time_series)
        freqs = np.fft.fftfreq(n)
        
        # Create mask for band
        mask = (np.abs(freqs) >= min_freq) & (np.abs(freqs) <= max_freq)
        
        # Apply mask
        filtered_fft = fft_result.copy()
        filtered_fft[~mask] = 0
        
        # Inverse FFT
        filtered = np.real(np.fft.ifft(filtered_fft))
        
        return filtered
    
    elif filter_type == 'fir':
        # Apply FIR filter using filtfilt for zero phase
        taps = band_filter['taps']
        filtered = signal.filtfilt(taps, [1.0], time_series)
        
        return filtered
    
    elif filter_type == 'iir':
        # Apply IIR filter using filtfilt for zero phase
        b = band_filter['b']
        a = band_filter['a']
        filtered = signal.filtfilt(b, a, time_series)
        
        return filtered
    
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")