import numpy as np
from operator import itemgetter


def simplify(loma):
    """?
    Parameters
    ----------
    loma: type
        description
    """
    simplified = []
    for l in loma:
        # align loma to it's position in the middle of the line
        pos =  l[int(len(l) / 2.)][0]
        strength = l[-1][1]
        simplified.append((pos, strength))
    return simplified


def get_prominences(pos_loma, alignment, rate=1):
    """?
    Parameters
    ----------
    pos_loma: list of ?
        Positive loma values
    labels: list of tuple (float, float, string)
        List of labels which are lists of 3 elements [start, end, description]
    """
    max_word_loma = []
    loma = simplify(pos_loma)
    for st, end in [(word.start(), word.end()) for word in alignment]:
        st *= rate
        end *= rate
        word_loma = []
        for l in loma:
            if l[0] >= st and l[0] <= end:
                word_loma.append(l)
        if len(word_loma) > 0:
            max_word_loma.append(sorted(word_loma, key=itemgetter(1))[-1])
        else:
            max_word_loma.append([st + (end - st) / 2., 0.])
    return max_word_loma


def get_boundaries(max_word_loma, boundary_loma, alignment):
    """get strongest lines of minimum amplitude between adjacent words' max lines"""
    boundary_loma = simplify(boundary_loma)
    max_boundary_loma = []
    st = 0
    end = 0
    for i in range(1, len(max_word_loma)):
        w_boundary_loma = []
        for l in boundary_loma:
            st = max_word_loma[i - 1][0]
            end = max_word_loma[i][0]
            if l[0] >= st and l[0] < end:
                if l[1] > 0:
                    w_boundary_loma.append(l)

        if len(w_boundary_loma) > 0:
            max_boundary_loma.append(
                sorted(w_boundary_loma, key=itemgetter(1))[-1])
        else:
            max_boundary_loma.append([st + (end - st) / 2, 0])

    # final boundary is not estimated
    max_boundary_loma.append((alignment.end(), 1))

    return max_boundary_loma


def _get_parent(child_index, parent_diff, parent_indices):
    """Private function to find the parent of the given child peak. At child peak index, follow the
    slope of parent scale upwards to find parent

    Parameters
    ----------
    child_index: int
        Index of the current child peak
    parent_diff: list of ?
        ?
    parent_indices: list of int ?
        Indices of available parents

    Returns
    _______
    int
    	The parent index or None if there is no parent
    """
    for i in range(0, len(parent_indices)):
        if parent_indices[i] > child_index:
            if parent_diff[int(child_index)] > 0:
                return parent_indices[i]
            else:
                if i > 0:
                    return parent_indices[i - 1]
                else:
                    return parent_indices[0]

    if len(parent_indices) > 0:
        return parent_indices[-1]


def get_loma(wavelet_matrix, scales, min_scale, max_scale):
    """Get the Line Of Maximum Amplitude (loma)

    Parameters
    ----------
    wavelet_matrix: matrix of float
        The wavelet matrix
    scales: list of int
        The list of scales
    min_scale: int
        The minimum scale
    max_scale: int
        The maximum scale

    Returns
    -------
    list of tuples
    	?

    Note
    ----
    change this so that one level is done in one chunk, not one parent.
    """
    min_peak = -10000. # minimum peak amplitude to consider. NOTE:this has no meaning unless scales normalized
    max_dist = 10 # how far in time to look for parent peaks. NOTE: frame rate and scale dependent

    # get peaks from the first scale
    peaks, indices = get_peaks(wavelet_matrix[min_scale], min_peak)

    loma = dict()
    root = dict()
    for i in range(0, len(peaks)):
        loma[indices[i]] = []

        # keep track of roots of each loma
        root[indices[i]] = indices[i]

    for i in range(min_scale + 1, max_scale):
        max_dist = np.sqrt(scales[i]) * 4

        # find peaks in the parent scale
        p_peaks, p_indices = get_peaks(wavelet_matrix[i], min_peak)
        parents = dict(zip(p_indices, p_peaks))

        # find a parent for each child peak
        children = dict()
        for p in p_indices:
            children[p] = []

        parent_diff = np.diff(wavelet_matrix[i], 1)
        for j in range(0, len(indices)):
            parent =_get_parent(indices[j], parent_diff, p_indices)
            if parent:
                if abs(parent - indices[j]) < max_dist and peaks[j] > min_peak:
                    children[parent].append([indices[j], peaks[j]])

        # for each parent, select max child
        peaks = []
        indices = []
        for p in children:
            if len(children[p]) > 0:
                maxi = sorted(children[p], key=itemgetter(1))[-1]
                indices.append(p)
                peaks.append(maxi[1] + parents[p])

                #append child to correct loma
                loma[root[maxi[0]]].append([maxi[0], maxi[1] + parents[p], i, p])
                root[p] = root[maxi[0]]

    sorted_loma = []
    for k in sorted(loma.keys()):
        if  len(loma[k]) > 0:
            sorted_loma.append(loma[k])

    return sorted_loma


def get_peaks(params, threshold=-10):
    """Find the peaks based on the given prosodic parameters.

    Parameters
    ----------
    params: ?
        Prosodic parameters
    threshold: int
        description

    Returns
    -------
    peaks: arraylike
        array of peak values and peak indices
    """
    indices = (np.diff(np.sign(np.diff(params))) < 0).nonzero()[0] + 1
    peaks = params[indices]
    return np.array([peaks[peaks > threshold], indices[peaks > threshold]])
