import penn

import torch

import emphases

###############################################################################
# Pitch variance method
###############################################################################


def infer(alignment, audio, sample_rate, gpu=None):
    """Compute per-word emphasis scores using pitch variance method"""
    # TODO
    mode = emphases.VARIANCE_MODE
    if mode == 'pitch':
        #Use penn to find pitches and periodicities
        pitch, periodicity = penn.from_audio(
            audio,
            sample_rate,
            hopsize = emphases.convert.samples_to_seconds(emphases.HOPSIZE),
            fmin = emphases.FMIN,
            fmax = emphases.FMAX,
            pad = True,
            interp_unvoiced_at = emphases.PENN_VOICED_THRESHOLD,
            gpu = gpu
        )

        #Initialize array of frames for full setnence
        all_pitches = torch.Tensor()
        word_variances = torch.zeros(len(alignment))
        
        #Iterate over words, get 95th percentile - 5th percentile of log pitches
        for i in range(0, len(alignment)):
            word = alignment[i]
            start_frame = int(emphases.convert.seconds_to_frames(word.start()))
            end_frame = int(emphases.convert.seconds_to_frames(word.end()))
            word_pitches = pitch[0][start_frame:end_frame]
            #Get 5th and 95th percentiles
            word_range_high = torch.quantile(word_pitches, 0.95)
            word_range_low = torch.quantile(word_pitches, 0.05)
            word_variances[i] = word_range_high - word_range_low
            #Add to array of all pitches
            all_pitches = torch.cat((all_pitches, word_pitches))
        
        sample_range_high = torch.quantile(all_pitches, 0.95)
        sample_range_low = torch.quantile(all_pitches, 0.05)
        
        variance_diff = word_variances - (sample_range_high - sample_range_low)
        #Put results between 0 and 1
        variance_diff = (variance_diff - variance_diff.min()) / variance_diff.max()
        return torch.reshape(variance_diff, (1, -1))
    elif mode == 'duration':
        total_phonemes = 0
        total_length = 0
        word_average = torch.zeros(len(alignment)) #Average per-word phoneme duration
        for i in range(len(alignment)):
            #Iterate through words
            word = alignment[i]
            word_sum = 0
            for phoneme in word: #Get duration of each phoneme, add to word and total length
                word_sum += phoneme.duration()
                total_length += phoneme.duration()
            word_average[i] = word_sum / len(word)
            total_phonemes += len(word)
        avg_length = total_length / total_phonemes #Gives sentence average per-phoneme duration
        result = torch.abs(word_average - avg_length)
        return torch.reshape(result, (1, -1))
    else:
        raise ValueError(f'Variance mode {mode} is not defined')
