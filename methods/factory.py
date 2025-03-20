def get_method_instance(method):
    """
    Factory function to get an instance of the specified method.
    
    Parameters
    ----------
    method : str or FrequencyBandAnalyzer
        Method name or instance
        
    Returns
    -------
    FrequencyBandAnalyzer
        Instance of the specified method
    """
    from .ufbd import UnsupervisedFrequencyBandDiscovery
    from .inchworm import InchwormFEBA
    from .hybrid import HybridDetector, AutoSelectAnalyzer
    from ..core.base import FrequencyBandAnalyzer
    
    # If already a method instance, return it
    if isinstance(method, FrequencyBandAnalyzer):
        return method
    
    # Create method instance based on string name
    if method == 'ufbd':
        return UnsupervisedFrequencyBandDiscovery()
    elif method == 'inchworm':
        return InchwormFEBA()
    elif method == 'hybrid':
        return HybridDetector()
    elif method == 'auto':
        return AutoSelectAnalyzer()
    else:
        raise ValueError(f"Unknown method: {method}")