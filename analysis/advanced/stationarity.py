# Advanced stationarity analysis with improved handling of statistical tests
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib.pyplot as plt
import warnings
from scipy import stats

# Define a custom warning for KPSS interpolation issues
class InterpolationWarning(UserWarning):
    pass

def modified_kpss(x, regression='c', nlags=None, store=False):
    """
    Modified KPSS test that handles extreme values better and suppresses warnings.
    
    Parameters
    ----------
    x : array_like
        The data series
    regression : str {'c', 'ct'}
        Indicates the null hypothesis for the KPSS test:
        'c' : The data is stationary around a constant (default)
        'ct' : The data is stationary around a trend
    nlags : int
        Number of lags to use in the estimation of sigma
    store : bool
        If True, then a result instance is returned additionally to
        the KPSS statistic (default is False).
        
    Returns
    -------
    kpss_stat : float
        The KPSS test statistic
    p_value : float
        The p-value for the test statistic
    lags : int
        The number of lags used
    crit_vals : dict
        The critical values for the test statistic
    """
    # Run standard KPSS but catch and handle the warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_result = kpss(x, regression=regression, nlags=nlags, store=store)
    
    # Extract results
    kpss_stat, p_value, lags, crit_vals = kpss_result
    
    # Handle extreme p-values
    if p_value == 0.01 and kpss_stat > crit_vals["1%"]:
        # If p-value is at the minimum and statistic exceeds critical value,
        # use a more conservative p-value approximation
        p_value = 0.001  # More conservative estimate
    
    elif p_value == 0.10 and kpss_stat < crit_vals["10%"]:
        # If p-value is at the maximum and statistic is below critical value,
        # use a more conservative p-value approximation
        p_value = 0.15  # More conservative estimate
    
    return kpss_stat, p_value, lags, crit_vals


def compute_hurst_exponent(time_series, max_lag=20):
    """
    Compute the Hurst exponent using the rescaled range (R/S) analysis.
    
    Parameters
    ----------
    time_series : np.ndarray
        Input time series
    max_lag : int
        Maximum lag to compute
        
    Returns
    -------
    float
        Hurst exponent (H):
        H < 0.5: anti-persistent series, likely stationary
        H = 0.5: random series (Brownian motion)
        H > 0.5: persistent series, likely non-stationary
    """
    # Convert to numpy array and ensure it's a float array for calculations
    ts = np.asarray(time_series, dtype=float)
    
    # Compute the array of the variances of the lagged differences
    lags = range(2, min(max_lag, len(ts) // 4))
    
    # Lists to store calculated values
    rs_values = []
    
    # Calculate R/S values for different lag values
    for lag in lags:
        # Split time series into chunks of size 'lag'
        chunks = len(ts) // lag
        if chunks < 1:
            continue
            
        # Array to store R/S values for each chunk
        rs_array = np.zeros(chunks)
        
        # Calculate R/S for each chunk
        for i in range(chunks):
            # Get the chunk
            chunk = ts[i * lag:(i + 1) * lag]
            
            # Mean-adjust and calculate cumulative deviate series
            mean_chunk = np.mean(chunk)
            adjusted = chunk - mean_chunk
            cumulative = np.cumsum(adjusted)
            
            # Calculate range and standard deviation
            r = np.max(cumulative) - np.min(cumulative)  # Range
            s = np.std(chunk)  # Standard deviation
            
            # Avoid division by zero
            if s == 0:
                rs_array[i] = 0
            else:
                rs_array[i] = r / s  # Rescaled range (R/S)
        
        # Average R/S values across chunks
        rs_values.append(np.mean(rs_array))
    
    # Convert to numpy arrays
    lags_array = np.array(lags)
    rs_array = np.array(rs_values)
    
    # Make sure we have enough valid data points
    if len(rs_array) <= 1:
        return 0.5  # Return default value for white noise
    
    # Remove any zeros or negative values before taking log
    valid_indices = (rs_array > 0) & (lags_array > 0)
    if np.sum(valid_indices) <= 1:
        return 0.5  # Return default value if not enough valid data
    
    # Calculate the Hurst Exponent as the slope of log-log plot
    slope, _, _, _, _ = stats.linregress(
        np.log10(lags_array[valid_indices]), 
        np.log10(rs_array[valid_indices])
    )
    
    return slope


def test_stationarity(time_series, method='combined', regression='ct'):
    """
    Enhanced test for time series stationarity using multiple approaches.
    
    Parameters
    ----------
    time_series : np.ndarray
        Input time series
    method : str
        Test method ('adf', 'kpss', 'combined', 'pp', 'hurst', 'ensemble')
        - 'adf': Augmented Dickey-Fuller test
        - 'kpss': KPSS test with improved handling
        - 'combined': Both ADF and KPSS (enhanced consensus approach)
        - 'pp': Phillips-Perron test
        - 'hurst': Hurst exponent analysis
        - 'ensemble': Combine multiple methods for a more robust assessment
    regression : str
        Type of regression for tests ('c', 'ct', etc.)
        
    Returns
    -------
    dict
        Dictionary with test results
    """
    results = {}
    
    # Augmented Dickey-Fuller test (null hypothesis: unit root exists = non-stationary)
    if method in ['adf', 'combined', 'ensemble']:
        try:
            adf_result = adfuller(time_series, regression=regression)
            results['adf_statistic'] = adf_result[0]
            results['adf_pvalue'] = adf_result[1]
            results['adf_stationary'] = adf_result[1] < 0.05  # Reject null = stationary
        except Exception as e:
            results['adf_error'] = str(e)
    
    # KPSS test (null hypothesis: series is stationary)
    if method in ['kpss', 'combined', 'ensemble']:
        try:
            # Use modified KPSS implementation
            kpss_stat, p_value, _, _ = modified_kpss(time_series, regression=regression)
            results['kpss_statistic'] = kpss_stat
            results['kpss_pvalue'] = p_value
            results['kpss_stationary'] = p_value > 0.05  # Fail to reject null = stationary
        except Exception as e:
            results['kpss_error'] = str(e)
    
    # Phillips-Perron test (similar to ADF but more robust to serial correlation)
    if method in ['pp', 'ensemble']:
        try:
            # Statsmodels doesn't have a direct PP test, fall back to ADF
            # In a production environment, you would implement the PP test
            # or use a library that provides it
            if 'adf_stationary' not in results:
                adf_result = adfuller(time_series, regression=regression)
                results['pp_statistic'] = adf_result[0]
                results['pp_pvalue'] = adf_result[1]
                results['pp_stationary'] = adf_result[1] < 0.05
                results['pp_fallback'] = 'Used ADF as fallback'
            else:
                results['pp_statistic'] = results['adf_statistic']
                results['pp_pvalue'] = results['adf_pvalue']
                results['pp_stationary'] = results['adf_stationary']
                results['pp_fallback'] = 'Used ADF results'
        except Exception as e:
            results['pp_error'] = str(e)
    
    # Hurst exponent analysis
    if method in ['hurst', 'ensemble']:
        try:
            hurst_exp = compute_hurst_exponent(time_series)
            results['hurst_exponent'] = hurst_exp
            # Hurst < 0.5: anti-persistent, likely stationary
            # Hurst = 0.5: random walk
            # Hurst > 0.5: persistent, likely non-stationary
            results['hurst_stationary'] = hurst_exp < 0.6  # Conservative threshold
        except Exception as e:
            results['hurst_error'] = str(e)
    
    # Combined result based on method
    if method == 'combined' and 'adf_stationary' in results and 'kpss_stationary' in results:
        # Both tests agree = strong evidence
        # ADF: reject null (p<0.05) = stationary
        # KPSS: fail to reject null (p>0.05) = stationary
        results['is_stationary'] = results['adf_stationary'] and results['kpss_stationary']
    
    elif method == 'ensemble':
        # Weighted voting from multiple tests
        stationary_votes = 0
        total_votes = 0
        
        if 'adf_stationary' in results:
            stationary_votes += 1 if results['adf_stationary'] else 0
            total_votes += 1
        
        if 'kpss_stationary' in results:
            stationary_votes += 1 if results['kpss_stationary'] else 0
            total_votes += 1
        
        if 'pp_stationary' in results:
            stationary_votes += 1 if results['pp_stationary'] else 0
            total_votes += 1
        
        if 'hurst_stationary' in results:
            stationary_votes += 1 if results['hurst_stationary'] else 0
            total_votes += 1
        
        if total_votes > 0:
            results['is_stationary'] = (stationary_votes / total_votes) >= 0.5
            results['stationarity_confidence'] = stationary_votes / total_votes
    
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
            results['variance_ratio'] = variance_ratio
            
            # If variance changes significantly, mark as non-stationary
            if variance_ratio > 3.0:  # Significant change in variance
                results['changing_variance'] = True
                # Reduce confidence in stationarity
                if 'is_stationary' in results and results['is_stationary']:
                    if 'stationarity_confidence' in results:
                        results['stationarity_confidence'] *= 0.7  # Reduce confidence
                    
                    # Only override in extreme cases
                    if variance_ratio > 5.0:
                        results['is_stationary'] = False
    except Exception as e:
        results['variance_test_error'] = str(e)
    
    return results


def test_band_components_stationarity(components, method='ensemble', regression='ct'):
    """
    Test stationarity of each frequency band component with enhanced methods.
    
    Parameters
    ----------
    components : dict
        Dictionary with band components
    method : str
        Test method (as in test_stationarity)
    regression : str
        Type of regression for tests
        
    Returns
    -------
    dict
        Dictionary with stationarity test results for each band
    """
    stationarity_results = {}
    
    for band_name, component in components.items():
        stationarity_results[band_name] = test_stationarity(
            component, method=method, regression=regression
        )
    
    return stationarity_results


def segment_stationarity_analysis(time_series, segments=10, method='ensemble'):
    """
    Analyze stationarity in different segments of the time series with enhanced methods.
    
    Parameters
    ----------
    time_series : np.ndarray
        Input time series
    segments : int
        Number of segments to analyze
    method : str
        Stationarity test method
        
    Returns
    -------
    dict
        Segmented stationarity analysis results
    """
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
    confidence = []
    for segment in segments:
        result = segment['result']
        is_stationary.append(result.get('is_stationary', False))
        confidence.append(result.get('stationarity_confidence', 0.5 if result.get('is_stationary', False) else 0))
    
    # Create colormap based on confidence levels
    colors = []
    for conf in confidence:
        if conf > 0.7:
            colors.append('darkgreen')  # High confidence stationary
        elif conf > 0.5:
            colors.append('lightgreen')  # Low confidence stationary
        elif conf > 0.3:
            colors.append('orange')  # Low confidence non-stationary
        else:
            colors.append('red')  # High confidence non-stationary
    
    # Create bar chart
    bars = ax.bar(np.arange(n_segments), [1] * n_segments, color=colors)
    
    # Add segment indices
    ax.set_xticks(np.arange(n_segments))
    ax.set_xticklabels([str(i+1) for i in range(n_segments)])
    ax.set_yticks([])
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='darkgreen', label='High confidence stationary'),
        Patch(facecolor='lightgreen', label='Low confidence stationary'),
        Patch(facecolor='orange', label='Low confidence non-stationary'),
        Patch(facecolor='red', label='High confidence non-stationary')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_title('Stationarity Analysis by Segment with Confidence Levels')
    ax.set_xlabel('Segment')
    
    return fig
