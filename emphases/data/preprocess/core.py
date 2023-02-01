import emphases
import penn
import torch
import pypar

###############################################################################
# Preprocess
###############################################################################


def datasets(datasets):
    """Preprocess datasets"""
    for dataset in datasets:
        data_directory = emphases.DATA_DIR / dataset
        cache_directory = emphases.CACHE_DIR / dataset

        # Get audio files
        audio_files = sorted(data_directory.rglob('*.wav'))

        # Get alignment files
        alignment_files = sorted(cache_directory.rglob('*.TextGrid'))

        # Get output filepaths for mels, loudness, prominence
        mel_files = [
             cache_directory / 'mels' / f'{file.stem}.pt'
             for file in audio_files]

        loudness_files = [
             cache_directory / 'loudness' / f'{file.stem}.pt'
             for file in audio_files]

        (cache_directory / 'prominence').mkdir(exist_ok=True, parents=True)
        prominence_files = [
             cache_directory / 'prominence' / f'{file.stem}.pt'
             for file in audio_files]

        # Preprocess mels
        emphases.data.preprocess.mels.from_files_to_files(
            audio_files,
            mel_files)

        # Preprocess loudness
        emphases.data.preprocess.loudness.from_files_to_files(
            audio_files,
            loudness_files)

        # Preprocess pitch, periodicity
        (cache_directory / 'pitch').mkdir(exist_ok=True, parents=True)
        pitch_files = [
            cache_directory / 'pitch' / f'{file.stem}'
            for file in audio_files]

        penn.from_files_to_files(
            audio_files,
            pitch_files,
            hopsize = emphases.convert.samples_to_seconds(emphases.HOPSIZE),
            fmin = emphases.FMIN,
            fmax = emphases.FMAX,
            pad = True,
            interp_unvoiced_at = emphases.PENN_VOICED_THRESHOLD,
            gpu = 1
        )

        # Preprocess prominence
        iterator = emphases.iterator(
            zip(audio_files, alignment_files, prominence_files),
            'Preparing prominence features',
            total=len(audio_files))

        for audio_file, alignment_file, save_file in iterator:
            prominence = emphases.baselines.prominence.infer(
                        pypar.Alignment(alignment_file),
                        emphases.load.audio(audio_file),
                        emphases.SAMPLE_RATE)
            prominence = torch.from_numpy(prominence)
            torch.save(prominence, save_file)


def from_audio(audio, alignment, sample_rate=emphases.SAMPLE_RATE, gpu=None):
    """Preprocess one audio file"""
    # Move to device (no-op if devices are the same)
    audio = audio.to('cpu' if gpu is None else f'cuda:{gpu}')

    # Preprocess mels
    features = emphases.data.preprocess.mels.from_audio(audio, sample_rate)

    if emphases.PITCH_FEATURE or emphases.PERIODICITY_FEATURE:
        # Preprocess pitch, periodicity
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
        
        if emphases.PITCH_FEATURE:
            pitch = torch.log2(pitch)[None, :].to(audio.device)
            features = torch.cat((features, pitch), dim=1)

        if emphases.PERIODICITY_FEATURE:
            periodicity = periodicity[None, :].to(audio.device)
            features = torch.cat((features, periodicity), dim=1)

    if emphases.LOUDNESS_FEATURE:
        loudness = emphases.data.preprocess.loudness.from_audio(audio, sample_rate)
        loudness = loudness[None, :].to(audio.device)
        features = torch.cat((features, loudness), dim=1)

    if emphases.PROMINENCE_FEATURE:
        prominence = emphases.baselines.prominence.infer(
            alignment,
            audio.cpu(),
            sample_rate)

        # Compute word bounds
        bounds = alignment.word_bounds(
            emphases.SAMPLE_RATE,
            emphases.HOPSIZE,
            silences=True)
        word_bounds = torch.cat(
            [torch.tensor(bound)[None] for bound in bounds]).T

        # Get center time of each word in frames
        word_centers = \
            word_bounds[0] + (word_bounds[1] - word_bounds[0]) / 2.

        # Get frame centers
        frame_centers = .5 + torch.arange(features.shape[-1])

        prominence = torch.from_numpy(prominence)
        # Need interpolation
        prominence = emphases.interpolate(
            frame_centers[None],
            word_centers[None],
            prominence).to(audio.device, dtype=torch.float)
        
        features = torch.cat((features, prominence[None, :]), dim=1)

    return features

