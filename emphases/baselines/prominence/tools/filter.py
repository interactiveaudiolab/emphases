from scipy.signal import butter, lfilter


def butter_bandpass(lowcut, highcut, fs, order=5):
    """Generate the butter bandpass filter

    For more details see scipy.signal.butter documentation

    Parameters
    ----------
    lowcut: int
        The low cut value
    highcut: type
        description
    fs: int
        Signal sample rate
    order: int
        Order of the butter fiter

    Returns
    -------
    b: arraylike
    	Numerator polynomial of the IIR filter
    a: arraylike
    	Denominator polynomial of the IIR filter
    """
    nyq = .5 * fs
    low = lowcut / nyq
    if highcut >= nyq * .95:
        highcut = nyq * .95
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Filter signal data using a butter filter type

    For more details see scipy.signal.butter and scipy.signal.lfilter documentation

    Parameters
    ----------
    data: arraylike
        An N-dimensional input array.
    lowcut: int
        The lowcut filtering value.
    highcut: type
        The highcut filtering value.
    fs: int
        The signal sample rate.
    order: int
        The order of the butter filter.

    Returns
    -------
    arraylike
    	An N-dimensional filtered array
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)
