from numpy import array, sqrt, pad, mean, pi

import pycwt as cwt


###########################################################################################
# Private routines
###########################################################################################


def _padded_cwt(params, dt, dj, s0, J, mother, padding_len):
    """Private function to compute a wavelet transform on padded data

    Parameters
    ----------
    params: arraylike
        The prosodic parameters.
    dt: ?
        ?
    dj: ?
        ?
    s0: ?
        ?
    J: ?
        ?
    mother: ?
        The mother wavelet.
    padding_len: int
        The padding length

    Returns
    -------
    wavelet_matrix: ndarray
    	The wavelet data resulting from the analysis
    scales: arraylike
    	The scale indices corresponding to the wavelet data
    freqs: ?
    	?
    coi: array
    	The cone of influence values
    fft: ?
    	?
    fftfreqs: ?
    	?
    """
    padded = pad(params, padding_len, mode='edge')
    wavelet_matrix, scales, freqs, coi, fft, fftfreqs = cwt.cwt(
        padded,
        dt,
        dj,
        s0,
        J,
        mother)
    wavelet_matrix = \
        wavelet_matrix[:, padding_len:len(wavelet_matrix[0]) - padding_len]
    return wavelet_matrix, scales, freqs, coi, fft, fftfreqs


def _zero_outside_coi(wavelet_matrix, freqs, rate=200):
    """Private function to set each elements outside of the Cone Of Influence (coi) to 0.

    Parameters
    ----------
    wavelet_matrix: type
        description
    freqs: type
        description
    """
    for i in range(0, wavelet_matrix.shape[0]):
        coi = int(1. / freqs[i] * rate)
        wavelet_matrix[i, :coi] = 0.
        wavelet_matrix[i, -coi:] = 0.
    return wavelet_matrix


def _scale_for_reconstruction(
    wavelet_matrix,
    scales,
    dj,
    dt,
    mother='mexican_hat',
    period=3):
    """ ?

    Parameters
    ----------
    wavelet_matrix: ndarray
    	The wavelet data resulting from the analysis
    scales: arraylike
    	The scale indices corresponding to the wavelet data
    dj: ?
        ?
    dt: ?
        ?
    mother: ?
        ?
    period: ?
        ?
    """
    scaled = array(wavelet_matrix)

    # mexican Hat
    c = dj / (3.541 * .867)

    if mother == 'morlet':
        cc = 1.83
        #periods 5 and 6 are correct, 3,4 approximate
        if period == 3:
            cc = 1.74
        if period == 4:
            cc = 1.1
        elif period == 5:
            cc = .9484
        elif period == 6:
            cc = .7784
        c = dj / (cc * pi ** (-.25))

    for i in range(0, len(scales)):
        scaled[i] *= c * sqrt(dt) / sqrt(scales[i])
        # substracting the mean should not be necessary?
        scaled[i] -= mean(scaled[i])

    return scaled


def cwt_analysis(
    params,
    mother_name='mexican_hat',
    num_scales=12,
    first_scale=None,
    scale_distance=1.,
    apply_coi=True,
    period=5,
    frame_rate=200):
    """Achieve the continous wavelet analysis of given parameters

    Parameters
    ----------
    params: arraylike
        The parameters to analyze.
    mother_name: string, optional
        The name of the mother wavelet [default: mexican_hat].
    num_scales: int, optional
        The number of scales [default: 12].
    first_scale: int, optional
        The width of the shortest scale
    scale_distance: float, optional
        The distance between scales [default: 1.0].
    apply_coi: boolean, optional
        Apply the Cone Of Influence (coi)
    period: int, optional
        The period of the mother wavelet [default: 5].
    frame_rate: int, optional
        The signal frame rate [default: 200].

    Returns
    -------
    wavelet_matrix: ndarray
    	The wavelet data resulting from the analysis
    scales: arraylike
    	The scale indices corresponding to the wavelet data
    """
    # setup wavelet transform
    dt = 1. / float(frame_rate)  # frame length

    if not first_scale:
        first_scale = dt # first scale, here frame length

    dj = scale_distance  # distance between scales in octaves
    J = num_scales #  number of scales

    mother = cwt.MexicanHat()

    if str.lower(mother_name) == 'morlet':
        mother = cwt.Morlet(period)

    wavelet_matrix, scales, freqs, *_ = _padded_cwt(
        params,
        dt,
        dj,
        first_scale,
        J,
        mother,
        400)
    wavelet_matrix = _scale_for_reconstruction(
        wavelet_matrix,
        scales,
        dj,
        dt,
        mother=mother_name,
        period=period)

    if apply_coi:
        wavelet_matrix = _zero_outside_coi(wavelet_matrix, freqs, frame_rate)

    return wavelet_matrix, scales, freqs
