# File: freqfinder/analysis/stationarity.py

import numpy as np
from statsmodels.tsa.stattools import adfuller
# kpss import moved to within the function to allow for fallback to modified version
import matplotlib.pyplot as plt
import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning

# Set a flag for using advanced stationarity methods
_USE_ADVANCED_METHODS = False

def use_advanced_stationarity_methods(enable=True):
    """
    Enable or disable advanced stationarity testing methods.
    
    Parameters
    ----------
    enable : bool
        Whether to enable advanced methods (True) or use standard methods (False)
    """
    global _USE_ADVANCED_METHODS
    _USE_ADVANCED_METHODS = enable
    
    if enable:
        try:
            # Attempt to import advanced methods to verify they're available
            from .advanced.stationarity import test_stationarity as adv_test
            return True
        except ImportError:
            warnings.warn(
                "Advanced stationarity methods not available. "
                "Reverting to standard methods."
            )
            _USE_ADVANCED_METHODS = False
            return False
    return True

def test_stationarity(time_series, method='both', regression='ct', use_advanced=None):
    """
    Test time series for stationarity using statistical tests.
    
    Parameters
    ----------
    time_series : np.ndarray
        Input time series
    method : str
        Test method ('adf', 'kpss', or 'both')
    regression : str
        Type of regression for tests ('c', 'ct', etc.)
    use_advanced : bool or None
        Whether to use advanced methods. If None, use the global setting.
        
    Returns
    -------
    dict
        Dictionary with test results
    """
    # Check if we should use advanced methods
    use_adv = _USE_ADVANCED_METHODS if use_advanced is None else use_advanced
    
    if use_adv:
        try:
            from .advanced.stationarity import test_stationarity as adv_test
            return adv_test(time_series, method='combined' if method == 'both' else method, 
                           regression=regression)
        except ImportError:
            warnings.warn("Advanced methods not available, falling back to standard")
            use_adv = False
    
    if not use_adv:
        results = {}
    
    # Augmented Dickey-Fuller test (null hypothesis: unit root exists = non-stationary)
    if method in ['adf', 'both']:
        try:
            adf_result = adfuller(time_series, regression=regression)
            results['adf_statistic'] = adf_result[0]
            results['adf_pvalue'] = adf_result[1]
            results['adf_stationary'] = adf_result[1] < 0.05  # Reject null = stationary
        except Exception as e:
            results['adf_error'] = str(e)
    
    # KPSS test (null hypothesis: series is stationary)
    if method in ['kpss', 'both']:
        try:
            # Try to import and use the modified_kpss from advanced module first
            try:
                from .advanced.stationarity import modified_kpss
                kpss_result = modified_kpss(time_series, regression=regression)
            except ImportError:
                # Fall back to standard kpss if advanced module is not available
                from statsmodels.tsa.stattools import kpss
                kpss_result = kpss(time_series, regression=regression)
                
            results['kpss_statistic'] = kpss_result[0]
            results['kpss_pvalue'] = kpss_result[1]
            results['kpss_stationary'] = kpss_result[1] > 0.05  # Fail to reject null = stationary
        except Exception as e:
            results['kpss_error'] = str(e)
    
    # Combined result (if both tests were run)
    if method == 'both' and 'adf_stationary' in results and 'kpss_stationary' in results:
        # Both tests agree = strong evidence
        results['is_stationary'] = results['adf_stationary'] and results['kpss_stationary']
    
    # Check for changing variance (non-stationarity indicator)
    try:
        # Split the series into segments and compare variance
        segments = 5
        segment_length = len(time_series) // segments
        if segment_length > 0:
            variances = []
            for i in range(segments):
                start = i * segment_length
                end = min((i + 1) * segment_length, len(time_series))
                segment = time_series[start:end]
                variances.append(np.var(segment))
            
            # Calculate variance of variances
            variance_ratio = max(variances) / (min(variances) + 1e-10)  # Avoid division by zero
            
            # If variance changes significantly, mark as non-stationary
            if variance_ratio > 3.0:  # Significant change in variance
                results['changing_variance'] = True
                # Override stationarity test results
                if 'is_stationary' in results:
                    results['is_stationary'] = False
    except Exception as e:
        results['variance_test_error'] = str(e)
    
    return results


def test_band_components_stationarity(components, regression='ct', use_advanced=None):
    """
    Test stationarity of each frequency band component.
    
    Parameters
    ----------
    components : dict
        Dictionary with band components
    regression : str
        Type of regression for tests
    use_advanced : bool or None
        Whether to use advanced methods. If None, use the global setting.
        
    Returns
    -------
    dict
        Dictionary with stationarity test results for each band
    """
    # Check if we should use advanced methods
    use_adv = _USE_ADVANCED_METHODS if use_advanced is None else use_advanced
    
    if use_adv:
        try:
            from .advanced.stationarity import test_band_components_stationarity as adv_test
            return adv_test(components, method='combined', regression=regression)
        except ImportError:
            warnings.warn("Advanced methods not available, falling back to standard")
            use_adv = False
    
    if not use_adv:
        stationarity_results = {}
    
    for band_name, component in components.items():
        stationarity_results[band_name] = test_stationarity(
            component, method='both', regression=regression
        )
    
    return stationarity_results


def segment_stationarity_analysis(time_series, segments=10, method='both', use_advanced=None):
    """
    Analyze stationarity in different segments of the time series.
    
    Parameters
    ----------
    time_series : np.ndarray
        Input time series
    segments : int
        Number of segments to analyze
    method : str
        Stationarity test method
    use_advanced : bool or None
        Whether to use advanced methods. If None, use the global setting.
        
    Returns
    -------
    dict
        Segmented stationarity analysis results
    """
    # Check if we should use advanced methods
    use_adv = _USE_ADVANCED_METHODS if use_advanced is None else use_advanced
    
    if use_adv:
        try:
            from .advanced.stationarity import segment_stationarity_analysis as adv_test
            return adv_test(time_series, segments=segments, 
                           method='combined' if method == 'both' else method)
        except ImportError:
            warnings.warn("Advanced methods not available, falling back to standard")
            use_adv = False
    
    if not use_adv:
        n = len(time_series)
    segment_length = n // segments
    
    results = []
    for i in range(segments):
        start = i * segment_length
        end = min((i + 1) * segment_length, n)
        
        segment_data = time_series[start:end]
        segment_result = test_stationarity(segment_data, method=method)
        
        results.append({
            'segment': i,
            'start': start,
            'end': end,
            'result': segment_result
        })
    
    return {
        'segments': results,
        'time_series_length': n,
        'segment_length': segment_length
    }


def plot_stationarity_results(stationarity_results, figsize=(10, 6)):
    """
    Plot stationarity test results for band components.
    
    Parameters
    ----------
    stationarity_results : dict
        Stationarity test results from test_band_components_stationarity
    figsize : tuple
        Figure size
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure with the plots
    """
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # Get bands and sort by name
    bands = sorted(stationarity_results.keys())
    
    # Extract test statistics and p-values
    adf_pvalues = [stationarity_results[band].get('adf_pvalue', np.nan) for band in bands]
    kpss_pvalues = [stationarity_results[band].get('kpss_pvalue', np.nan) for band in bands]
    
    # Stationarity results
    is_stationary = [stationarity_results[band].get('is_stationary', False) for band in bands]
    
    # Plot ADF test p-values
    ax = axes[0]
    # Filter out NaN/Inf values for plotting and limits
    valid_adf_pvalues = [p for p in adf_pvalues if not np.isnan(p) and not np.isinf(p)]
    
    # Use default p-value range if no valid values
    if not valid_adf_pvalues:
        valid_adf_pvalues = [0.05]
    
    bars = ax.bar(
        np.arange(len(bands)),
        adf_pvalues,
        color=['green' if not np.isnan(p) and p < 0.05 else 'red' for p in adf_pvalues]
    )
    
    ax.axhline(0.05, color='black', linestyle='--', alpha=0.7, label='p=0.05')
    ax.set_xticks(np.arange(len(bands)))
    ax.set_xticklabels(bands, rotation=45, ha='right')
    ax.set_ylabel('p-value')
    ax.set_title('ADF Test (p<0.05 indicates stationarity)')
    ax.set_ylim(0, max(max(valid_adf_pvalues) * 1.1, 0.1))
    
    # Plot KPSS test p-values
    ax = axes[1]
    # Filter out NaN/Inf values for plotting and limits
    valid_kpss_pvalues = [p for p in kpss_pvalues if not np.isnan(p) and not np.isinf(p)]
    
    # Use default p-value range if no valid values
    if not valid_kpss_pvalues:
        valid_kpss_pvalues = [0.05]
    
    bars = ax.bar(
        np.arange(len(bands)),
        kpss_pvalues,
        color=['green' if not np.isnan(p) and p > 0.05 else 'red' for p in kpss_pvalues]
    )
    
    ax.axhline(0.05, color='black', linestyle='--', alpha=0.7, label='p=0.05')
    ax.set_xticks(np.arange(len(bands)))
    ax.set_xticklabels(bands, rotation=45, ha='right')
    ax.set_ylabel('p-value')
    ax.set_title('KPSS Test (p>0.05 indicates stationarity)')
    ax.set_ylim(0, max(max(valid_kpss_pvalues) * 1.1, 0.1))
    
    plt.tight_layout()
    return fig


def plot_segment_stationarity(segment_results, figsize=(12, 6)):
    """
    Plot stationarity results for different segments.
    
    Parameters
    ----------
    segment_results : dict
        Results from segment_stationarity_analysis
    figsize : tuple
        Figure size
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure with the plots
    """
    segments = segment_results['segments']
    n_segments = len(segments)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract stationarity results for each segment
    is_stationary = []
    for segment in segments:
        result = segment['result']
        is_stationary.append(result.get('is_stationary', False))
    
    # Create heatmap-like visualization
    cmap = plt.cm.RdYlGn  # Red (non-stationary) to Green (stationary)
    im = ax.imshow(
        [is_stationary], 
        aspect='auto',
        cmap=cmap,
        extent=[0, len(is_stationary), 0, 1]
    )
    
    # Add segment indices
    ax.set_xticks(np.arange(n_segments) + 0.5)
    ax.set_xticklabels([str(i+1) for i in range(n_segments)])
    ax.set_yticks([])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', ticks=[0, 1])
    cbar.set_ticklabels(['Non-stationary', 'Stationary'])
    
    ax.set_title('Stationarity Analysis by Segment')
    ax.set_xlabel('Segment')
    
    return fig