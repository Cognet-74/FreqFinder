# File: freqfinder/core/visualization.py

import numpy as np
import matplotlib.pyplot as plt



def plot_detected_bands(bands, spectrum=None, ax=None, method_name=None):
    """
    Plot detected frequency bands with optional spectrum overlay.
    
    Parameters
    ----------
    bands : dict
        Band information in standardized format
    spectrum : tuple, optional
        Tuple of (frequencies, power) for spectrum overlay
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    method_name : str, optional
        Name of detection method for title
        
    Returns
    -------
    matplotlib.axes.Axes
        Axes with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot spectrum if provided
    if spectrum is not None:
        frequencies, power = spectrum
        ax.plot(frequencies, power, 'k-', alpha=0.5, label='Power Spectrum')
    
    # Create colormap for bands
    colors = plt.cm.tab10.colors
    
    # Plot each band
    for i, (band_name, band_info) in enumerate(bands.items()):
        color = colors[i % len(colors)]
        
        # Ensure we have central_freq - calculate if missing
        if 'central_freq' not in band_info:
            # Calculate central frequency as the midpoint
            band_info['central_freq'] = (band_info['min_freq'] + band_info['max_freq']) / 2
        
        # Shade band region
        ax.axvspan(
            band_info['min_freq'], 
            band_info['max_freq'], 
            alpha=0.2, 
            color=color,
            label=band_name
        )
        
        # Mark central frequency
        ax.axvline(
            band_info['central_freq'], 
            color=color, 
            linestyle='--', 
            alpha=0.7
        )
    
    # Set title and labels
    title = "Detected Frequency Bands"
    if method_name:
        title += f" ({method_name})"
        
    ax.set_title(title)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Power' if spectrum is not None else 'Bands')
    ax.legend(loc='best')
    
    return ax


def plot_band_components(time_series, components, stationarity_results=None, sampling_rate=1.0, figsize=(12, 8)):
    """
    Plot original time series and band components with stationarity information.
    
    Parameters
    ----------
    time_series : np.ndarray
        Original time series
    components : dict
        Dictionary with band components
    stationarity_results : dict, optional
        Stationarity test results
    sampling_rate : float
        Sampling rate in Hz
    figsize : tuple
        Figure size
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure with the plots
    """
    n_bands = len(components)
    fig, axes = plt.subplots(n_bands + 1, 1, figsize=figsize, sharex=True)
    
    # Plot original time series
    t = np.arange(len(time_series)) / sampling_rate
    axes[0].plot(t, time_series, 'k-')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Original Time Series')
    axes[0].grid(True)
    
    # Get colors
    colors = plt.cm.tab10.colors
    
    # Plot each band component
    for i, (band_name, component) in enumerate(components.items(), 1):
        color = colors[(i-1) % len(colors)]
        
        # Plot component
        axes[i].plot(t, component, color=color)
        axes[i].set_ylabel('Amplitude')
        
        # Add stationarity information if available
        if stationarity_results is not None and band_name in stationarity_results:
            stat_result = stationarity_results[band_name]
            is_stationary = stat_result.get('is_stationary', None)
            
            if is_stationary is not None:
                stat_text = "Stationary" if is_stationary else "Non-stationary"
                title = f"{band_name} ({stat_text})"
            else:
                title = band_name
        else:
            title = band_name
            
        axes[i].set_title(title)
        axes[i].grid(True)
    
    # Set common x label
    axes[-1].set_xlabel('Time (s)')
    
    plt.tight_layout()
    return fig


def plot_time_frequency(time_series, bands=None, sampling_rate=1.0, 
                       figsize=(10, 6), method=None):
    """
    Plot time-frequency representation with optional band overlay.
    
    Parameters
    ----------
    time_series : np.ndarray
        Input time series data
    bands : dict, optional
        Band information for overlay
    sampling_rate : float
        Sampling rate in Hz
    figsize : tuple
        Figure size
    method : str or callable, optional
        Method for computing time-frequency representation
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure with the plot
    """
    from scipy import signal
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Default to spectrogram if no method specified
    if method is None:
        f, t, Sxx = signal.spectrogram(
            time_series, 
            fs=sampling_rate, 
            nperseg=min(256, len(time_series)//10),
            noverlap=min(128, len(time_series)//20)
        )
        
        # Plot spectrogram
        pcm = ax.pcolormesh(
            t, f, 10 * np.log10(Sxx + 1e-10), 
            shading='gouraud',
            cmap='viridis'
        )
        
        # Add colorbar
        cbar = plt.colorbar(pcm, ax=ax)
        cbar.set_label('Power/Frequency (dB/Hz)')
    else:
        # Use provided method
        if callable(method):
            t, f, Sxx = method(time_series, sampling_rate)
            pcm = ax.pcolormesh(t, f, Sxx, shading='gouraud', cmap='viridis')
            plt.colorbar(pcm, ax=ax)
        else:
            raise ValueError(f"Method must be callable, got {type(method)}")
    
    # Overlay bands if provided
    if bands is not None:
        colors = plt.cm.tab10.colors
        
        for i, (band_name, band_info) in enumerate(bands.items()):
            color = colors[i % len(colors)]
            min_freq = band_info['min_freq']
            max_freq = band_info['max_freq']
            
            # Draw horizontal lines at band boundaries
            ax.axhline(min_freq, color=color, linestyle='--', alpha=0.7)
            ax.axhline(max_freq, color=color, linestyle='--', alpha=0.7)
            
            # Add band name
            ax.text(
                t[-1] * 1.01, 
                (min_freq + max_freq) / 2,
                band_name,
                color=color,
                va='center'
            )
    
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Time-Frequency Representation')
    
    return fig