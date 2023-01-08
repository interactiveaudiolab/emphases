import numpy as np

import scipy.signal

import emphases


def _get_f0(spec, energy, min_hz, max_hz, thresh, sil_thresh):
    """
    return frequency bin with maximum energy, if it is over given threshold
    and overall energy of the frame is over silence threshsold
    otherwise return 0 (unvoiced)
    """
    cand = int(min_hz) + np.argmax(spec[int(min_hz):int(max_hz)])
    if spec[cand] > thresh and energy > sil_thresh:
        if cand > 2 * min_hz and spec[int(round(cand / 2.))] > spec[cand] * .5:
            return int(round(cand / 2.))
        else:
            return cand
    return 0


def _track_pitch(
    pic,
    min_hz=50,
    max_hz=450,
    thresh=.1,
    energy_thresh=1.):
    """
    extract pitch contour from time-frequency image
    bin with maximum energy / frame is chosen as a first f0 estimate,
    following with refinement steps based on the assumption of continuity of the pitch track
    """
    pitch = np.zeros(pic.shape[0])

    # calc energy threshold for voicing
    log_energy = np.log(np.sum(pic, axis=1))
    energy_thresh = \
        np.min(emphases.baselines.prominence.smooth_and_interp.smooth(log_energy, 20)) + energy_thresh
    pic_smooth = pic * scipy.ndimage.gaussian_filter(pic, [2, 5])

    # find frequency bins with max_energy
    for i in range(0, pic_smooth.shape[0]):
        pitch[i] = _get_f0(
            pic_smooth[i],
            log_energy[i],
            min_hz,
            max_hz,
            thresh,
            energy_thresh)

    # second pass with soft constraints
    n_iters = 3
    from scipy.signal import gaussian

    for iter in range(0, n_iters):
        smoothed = emphases.baselines.prominence.f0_processing.process(pitch)
        smoothed = emphases.baselines.prominence.smooth_and_interp.smooth(smoothed, int(200. / (iter + 1.)))

        # gradually thightening gaussian window centered on current estimate to softly constrain next iteration
        win_len = 800
        g_window = gaussian(win_len, int(np.mean(smoothed) * (1. / (iter + 1.) ** 2)))

        for i in range(0, pic.shape[0]):
            window = np.zeros(len(pic_smooth[i]))
            st = int(np.max((0, int(smoothed[i] - win_len))))
            end = int(np.min((int(smoothed[i] + win_len * .5), win_len - st)))
            window[st:end] = g_window[win_len - end:]
            pitch[i] = _get_f0(
                pic_smooth[i] * window, log_energy[i],
                min_hz,
                max_hz,
                thresh,
                energy_thresh)

    return pitch


def _assign_to_bins(pic, freqs, mags):
    for i in range(1, freqs.shape[0] - 1):
        for j in range(0, freqs.shape[1]):
            try:
                pic[j, int(freqs[i, j])] += mags[i, j]
            except:
                pass


def inst_freq_pitch(
    wav_form,
    fs,
    min_hz=emphases.FMIN,
    max_hz=emphases.FMAX,
    voicing_thresh=emphases.VOICED_THRESHOLD,
    target_rate=200):
    """Extract speech f0 using the continuous wavelet transform"""
    voicing_thresh = (voicing_thresh - 50.) / 100.
    sample_rate = 4000
    tmp_wav_form = emphases.baselines.prominence.resample(wav_form, fs, sample_rate)
    tmp_wav_form = emphases.baselines.prominence.normalize(tmp_wav_form)

    DEC = int(round(sample_rate / target_rate))

    pic = np.zeros(
        shape=(int(len(tmp_wav_form) / float(DEC)), int(sample_rate / 4.)))

    # use continuous wavelet transform to get instantenous frequencies
    # integrate analyses with morlet mother wavelets with period = 5 for
    # good time and frequency resolution
    # setup wavelet
    s0 = 2. / sample_rate
    dj = .05 # 20 scales per octave
    J = 120  # six octaves
    dt = 1. / sample_rate
    periods = [5]
    for p in periods:
        wavelet_matrix, *_ = emphases.baselines.prominence.cwt_utils.cwt_analysis(
            tmp_wav_form,
            mother_name='morlet',
            first_scale=s0,
            num_scales=J,
            scale_distance=dj,
            apply_coi=False,
            period=p,
            frame_rate=sample_rate)

        # hilbert transform
        phase = np.unwrap(np.angle(wavelet_matrix), axis=1)
        freqs =  np.abs((np.gradient(phase, dt)[1]) / (2. * np.pi))

        freqs = scipy.signal.decimate(freqs, DEC, zero_phase=True)
        mags = scipy.signal.decimate(abs(wavelet_matrix), DEC, zero_phase=True)

        # normalize magnitudes
        mags = (mags - mags.min()) / mags.ptp()

        # construct time-frequency image
        _assign_to_bins(pic, freqs, mags)

    # perform frequency domain autocorrelation to enhance f0
    pic = scipy.ndimage.filters.gaussian_filter(pic, [1, 1])
    length = np.min((max_hz * 3, pic.shape[1])).astype(int)

    for i in range(0, pic.shape[0]):
        acorr1 = np.correlate(pic[i, :length], pic[i, :length], mode='same')
        pic[i, :int(length / 2.)] *= acorr1[int(len(acorr1) / 2.):]

    return _track_pitch(pic, min_hz, max_hz, voicing_thresh)
