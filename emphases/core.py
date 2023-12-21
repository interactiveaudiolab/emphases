import contextlib
import functools
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Type, Union

import huggingface_hub
import pyfoal
import pypar
import torch
import torchutil
import torchaudio

import emphases


###############################################################################
# Emphasis annotation API
###############################################################################


def from_file(
    text_file: Union[str, bytes, os.PathLike],
    audio_file: Union[str, bytes, os.PathLike],
    checkpoint: Optional[Union[str, bytes, os.PathLike]] = None,
    batch_size: Optional[int] = None,
    gpu: Optional[int] = None
) -> Tuple[Type[pypar.Alignment], torch.Tensor]:
    """Produce emphasis scores for each word for files on disk

    Args:
        text_file: The speech transcript (.txt) or alignment (.TextGrid) file
        audio_file: The speech waveform audio file
        checkpoint: The model checkpoint to use for inference
        batch_size: The maximum number of frames per batch
        gpu: The index of the gpu to run inference on

    Returns:
        alignment: The forced phoneme alignment
        scores: The float-valued emphasis scores for each word
    """
    # Load audio
    audio = emphases.load.audio(audio_file)

    if str(text_file).endswith('.TextGrid'):

        # Load alignment
        alignment = pypar.Alignment(text_file)

        # Detect emphases
        return from_alignment_and_audio(
            alignment,
            audio,
            emphases.SAMPLE_RATE,
            checkpoint,
            batch_size,
            gpu)

    else:

        # Load text
        with open(text_file, encoding='utf-8') as file:
            text = file.read()

        # Detect emphases
        return from_text_and_audio(
            text,
            audio,
            emphases.SAMPLE_RATE,
            checkpoint,
            batch_size,
            gpu)


def from_file_to_file(
    text_file: List[Union[str, bytes, os.PathLike]],
    audio_file: List[Union[str, bytes, os.PathLike]],
    output_prefix: Optional[List[Union[str, bytes, os.PathLike]]] = None,
    checkpoint: Optional[Union[str, bytes, os.PathLike]] = None,
    batch_size: Optional[int] = None,
    gpu: Optional[int] = None
) -> None:
    """Produce emphasis scores for each word for files on disk and save to disk

    Args:
        text_file: The speech transcript (.txt) or alignment (.TextGrid) file
        audio_file: The speech waveform audio file
        output_prefix: The output prefix. Defaults to text file stem.
        checkpoint: The model checkpoint to use for inference
        batch_size: The maximum number of frames per batch
        gpu: The index of the gpu to run inference on
    """
    if output_prefix is None:
        output_prefix = text_file.stem

    # Detect emphases
    results = from_file(
        text_file,
        audio_file,
        checkpoint,
        batch_size,
        gpu)
    if text_file.name.endswith('.txt'):
        alignment, prominence = results
    else:
        alignment = pypar.Alignment(text_file)
        prominence = results

    # Save results
    alignment.save(f'{output_prefix}.TextGrid')
    torch.save(results.cpu(), f'{output_prefix}.pt')


def from_files_to_files(
    text_files: List[Union[str, bytes, os.PathLike]],
    audio_files: List[Union[str, bytes, os.PathLike]],
    output_prefixes: Optional[List[Union[str, bytes, os.PathLike]]] = None,
    checkpoint: Optional[Union[str, bytes, os.PathLike]] = None,
    batch_size: Optional[int] = None,
    gpu: Optional[int] = None
) -> None:
    """Produce emphasis scores for each word for many files and save to disk

    Args:
        text_file: The speech transcript (.txt) or alignment (.TextGrid) files
        audio_files: The corresponding speech audio files
        output_prefixes: The output files. Defaults to text file stems.
        checkpoint: The model checkpoint to use for inference
        batch_size: The maximum number of frames per batch
        gpu: The index of the gpu to run inference on
    """
    # Set default output path
    if output_prefixes is None:
        output_prefixes = [file.stem for file in text_files]

    # Batch forced alignments to improve performance via multiprocessing
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)

        # Get files to force-align
        try:
            text_filtered, audio_filtered = zip(*[
                (text, audio) for text, audio in zip(text_files, audio_files)
                if str(text).endswith('.txt')])

            # Get location to save files
            output_filtered = [
                directory / Path(file).with_suffix('.TextGrid')
                for file in text_filtered]

            # Force align
            pyfoal.from_files_to_files(
                text_filtered,
                audio_filtered,
                output_filtered,
                aligner='p2fa')

        except ValueError:
            pass

        # Update filenames for emphasis detection
        text_files, audio_files = zip(*[
            (text, audio) if str(text).endswith('TextGrid')
            else (directory / Path(text).with_suffix('.TextGrid'), audio)
            for text, audio in zip(text_files, audio_files)])

        # Detect emphases
        annotation_fn = functools.partial(
            from_file_to_file,
            checkpoint=checkpoint,
            batch_size=batch_size,
            gpu=gpu)
        for files in torchutil.iterator(
            zip(text_files, audio_files, output_prefixes),
            emphases.CONFIG,
            total=len(text_files)
        ):
            annotation_fn(*files)


def from_text_and_audio(
    text: str,
    audio: torch.Tensor,
    sample_rate: int,
    checkpoint: Optional[Union[str, bytes, os.PathLike]] = None,
    batch_size: Optional[int] = None,
    gpu: Optional[int] = None
) -> Tuple[Type[pypar.Alignment], torch.Tensor]:
    """Produce emphasis scores for each word

    Args:
        text: The speech transcript
        audio: The speech waveform
        sample_rate: The audio sampling rate
        checkpoint: The model checkpoint to use for inference
        batch_size: The maximum number of frames per batch
        gpu: The index of the gpu to run inference on

    Returns:
        alignment: The forced phoneme alignment
        scores: The float-valued emphasis scores for each word
    """
    # Get word alignment
    alignment = pyfoal.from_text_and_audio(
        text,
        audio,
        sample_rate,
        aligner='p2fa')

    # Infer
    scores = from_alignment_and_audio(
        alignment,
        audio,
        sample_rate,
        checkpoint,
        batch_size,
        gpu)

    return alignment, scores


def from_alignment_and_audio(
    alignment: pypar.Alignment,
    audio: torch.Tensor,
    sample_rate: int,
    checkpoint: Optional[Union[str, bytes, os.PathLike]] = None,
    batch_size: Optional[int] = None,
    gpu: Optional[int] = None
) -> Tuple[Type[pypar.Alignment], torch.Tensor]:
    """Produce emphasis scores for each word

    Args:
        alignment: The forced phoneme alignment
        audio: The speech waveform
        sample_rate: The audio sampling rate
        checkpoint: The model checkpoint to use for inference
        batch_size: The maximum number of frames per batch
        gpu: The index of the gpu to run inference on

    Returns:
        scores: The float-valued emphasis scores for each word
    """
    # Neural method
    if emphases.METHOD == 'neural':

        scores = []

        # Preprocess audio
        for features, word_bounds in preprocess(
            alignment,
            audio,
            sample_rate,
            batch_size,
            gpu
        ):

            # Infer
            logits = infer(features, word_bounds, checkpoint).detach()[0]

            # Postprocess
            scores.append(postprocess(logits))

        # Concatenate results
        return torch.cat(scores, 1)

    # Prominence method
    if emphases.METHOD == 'prominence':
        return emphases.baselines.prominence.infer(
            alignment,
            audio,
            sample_rate)

    # Pitch variance method
    if emphases.METHOD == 'pitch-variance':
        return emphases.baselines.pitch_variance.infer(
            alignment,
            audio,
            sample_rate,
            gpu)

    # Duration variance method
    if emphases.METHOD == 'duration-variance':
        return emphases.baselines.duration_variance.infer(alignment)

    raise ValueError(
        f'Emphasis annotation method {emphases.METHOD} is not defined')


###############################################################################
# Inference steps
###############################################################################


def infer(features, word_bounds, checkpoint=None):
    """Perform model inference to annotate emphases of each word"""
    # Maybe cache model
    if (
        not hasattr(infer, 'model') or
        infer.checkpoint != checkpoint or
        infer.device_type != features.device.type
    ):
        # Initialize model
        model = emphases.Model()

        # Maybe download from HuggingFace
        if checkpoint is None:
            checkpoint = huggingface_hub.hf_hub_download(
                'maxrmorrison/emphases',
                'model.pt')

        # Load from disk
        infer.model, *_ = torchutil.checkpoint.load(checkpoint, model)
        infer.checkpoint = checkpoint
        infer.device_type = features.device.type

    # Move model to correct device (no-op if devices are the same)
    infer.model = infer.model.to(features.device)

    # Use full sequence lengths
    frame_lengths = torch.tensor(
        [features.shape[-1]],
        dtype=torch.long,
        device=features.device)
    word_lengths = torch.tensor(
        [word_bounds.shape[-1]],
        dtype=torch.long,
        device=features.device)

    # Infer
    with emphases.inference_context(infer.model):
        return infer.model(features, frame_lengths, word_bounds, word_lengths)


def postprocess(logits):
    """Postprocess network output"""
    if emphases.METHOD == 'neural':
        if emphases.LOSS == 'bce':
            return torch.sigmoid(logits)
        elif emphases.LOSS == 'mse':
            return torch.clamp(logits, 0., 1.)
    return logits


def preprocess(
    alignment,
    audio,
    sample_rate=emphases.SAMPLE_RATE,
    batch_size=None,
    gpu=None):
    """Convert audio to model input"""
    # Resample
    if sample_rate != emphases.SAMPLE_RATE:
        audio = resample(audio, sample_rate)

    # Pad audio and get total number of frames
    padding = int((emphases.WINDOW_SIZE - emphases.HOPSIZE) / 2)
    audio = torch.nn.functional.pad(audio, (padding, padding))
    total_frames = int(audio.shape[-1] / emphases.HOPSIZE)

    # Default to running all frames in a single batch
    batch_size = total_frames if batch_size is None else batch_size

    # Generate batches
    start = 0
    while start < len(alignment):

        # Accumulate enough frames for this batch
        frames = 0.
        end = start + 1
        while end < len(alignment):

            # Update frames
            frames += emphases.convert.seconds_to_frames(
                alignment[end - 1].duration())

            # Stop if we've accumulated enough frames
            if int(frames) > batch_size:
                break

            end += 1

        # Slice alignment
        batch_alignment = alignment[start:end]

        # Compute word bounds
        bounds = batch_alignment.word_bounds(
            emphases.SAMPLE_RATE,
            emphases.HOPSIZE,
            silences=True)
        batch_word_bounds = torch.cat(
            [torch.tensor(bound)[None] for bound in bounds]).T[None]

        # Slice audio at frame boundaries
        start_sample = int(emphases.convert.frames_to_samples(
            int(emphases.convert.seconds_to_frames(
                alignment[start].start()))))
        end_sample = int(emphases.convert.frames_to_samples(
            int(emphases.convert.seconds_to_frames(
                alignment[end - 1].end()))))
        batch_audio = audio[:, start_sample:end_sample]

        try:

            # Preprocess audio
            batch_features = emphases.data.preprocess.from_audio(
                batch_audio,
                gpu)

            # Run inference
            yield batch_features, batch_word_bounds

        # Handle residual frame
        except RuntimeError:
            pass

        # Update start word
        start = end


###############################################################################
# Word and frame resolution resampling
###############################################################################


def downsample(xs, word_bounds, word_lengths):
    """Interpolate from frame to word resolution"""
    # Average resampling
    if emphases.DOWNSAMPLE_METHOD in ['average', 'max', 'sum']:

        # Allocate memory for word resolution sequence
        result = torch.zeros(
            (xs.shape[0], xs.shape[1], word_lengths.max().item()),
            dtype=xs.dtype,
            device=xs.device)

        # Iterate over batch
        for i, (x, bounds, length) in enumerate(
            zip(xs, word_bounds, word_lengths)
        ):

            # Iterate over words
            for j in range(length):

                # Get word bounds
                start, end = bounds[0, j], bounds[1, j]

                # Downsample
                if emphases.DOWNSAMPLE_METHOD == 'average':
                    result[i, :, j] = x[:, start:end].mean(dim=1)
                elif emphases.DOWNSAMPLE_METHOD == 'max':
                    result[i, :, j] = x[:, start:end].max(dim=1).values
                elif emphases.DOWNSAMPLE_METHOD == 'sum':
                    result[i, :, j] = x[:, start:end].sum(dim=1)

        return result

    # Centerpoint resampling
    if emphases.DOWNSAMPLE_METHOD == 'center':

        # Downsample to words via index selection
        indices = (word_bounds[:, 0] + word_bounds[:, 1]) // 2
        return xs.transpose(1, 2)[
            torch.arange(xs.shape[0])[:, None],
            indices
        ].transpose(1, 2)

    raise ValueError(
        f'Interpolation method {emphases.DOWNSAMPLE_METHOD} is not defined')


def upsample(xs, word_bounds, word_lengths, frame_lengths):
    """Interpolate from word to frame resolution"""
    result = torch.zeros(
        (xs.shape[0], xs.shape[1], frame_lengths.max()),
        dtype=xs.dtype,
        device=xs.device)
    for i, (x, word_bound, word_length, frame_length) in enumerate(
        zip(xs, word_bounds, word_lengths, frame_lengths)
    ):

        # Truncate to valid sequence
        x = x[..., :word_length]
        word_bound = word_bound[..., :word_length]

        # Get center time of each word in frames
        word_times = (
            word_bound[0] + (word_bound[1] - word_bound[0]) / 2.)[None]

        # Get frame centers
        frame_times = .5 + torch.arange(frame_length, device=xs.device)[None]

        # Single value edge case
        if x.shape[1] == 1:
            result[i, :, :frame_length] = x[0]

        # Linear interpolation
        elif emphases.UPSAMPLE_METHOD == 'linear':

            # Compute slope and intercept at original times
            slope = (
                (x[:, 1:] - x[:, :-1]) /
                (word_times[:, 1:] - word_times[:, :-1]))
            intercept = x[:, :-1] - slope.mul(word_times[:, :-1])

            # Compute indices at which we evaluate points
            indices = torch.sum(
                torch.ge(frame_times[:, :, None], word_times[:, None, :]),
                -1
            ) - 1
            indices = torch.clamp(indices, 0, slope.shape[-1] - 1)

            # Compute index into parameters
            line_idx = torch.linspace(
                0,
                indices.shape[0],
                1,
                device=indices.device).to(torch.long)
            line_idx = line_idx.expand(indices.shape)

            # Interpolate
            result[i, :, :frame_length] = (
                slope[line_idx, indices].mul(frame_times) +
                intercept[line_idx, indices])

        # Nearest neighbors interpolation
        elif emphases.UPSAMPLE_METHOD == 'nearest':

            # Compute indices at which we evaluate points
            indices = torch.sum(
                torch.ge(frame_times[:, :, None], word_times[:, None, :]),
                -1
            ) - 1
            indices = torch.clamp(indices, 0, word_times.shape[-1] - 1)

            # Get nearest score
            result[i, :, :frame_length] = torch.index_select(x, 1, indices[0])

        else:
            raise ValueError(
                f'Interpolation method {emphases.UPSAMPLE_METHOD} ' +
                'is not defined')

    return result


###############################################################################
# Word segmentation
###############################################################################


def segment(xs, word_bounds, word_lengths):
    """Convert acoustic features to word segments"""
    # Get maximum word length
    max_length = (word_bounds[:, 1] - word_bounds[:, 0]).max()

    # Allocate memory
    batch = word_bounds.shape[0] * word_bounds.shape[2]
    result = torch.zeros(
        (batch, xs.shape[1], max_length),
        dtype=xs.dtype,
        device=xs.device)
    result_bounds = torch.zeros(
        (batch, 2, 1),
        dtype=torch.long,
        device=xs.device)
    result_lengths = torch.zeros((batch,), dtype=torch.long, device=xs.device)

    # Iterate over batch
    for i, (x, bounds, words) in enumerate(zip(xs, word_bounds, word_lengths)):

        # Iterate over words
        for j in range(bounds.shape[1]):
            k = min(j, words - 1)

            # Get word bounds
            start, end = bounds[0, k], bounds[1, k]

            # Chunk
            frames = end - start
            index = i * bounds.shape[1] + j
            result[index, :, :frames] = x[:, start:end]
            result_bounds[index, 1, 0] = frames
            result_lengths[index] = frames

    return result, result_bounds, result_lengths


###############################################################################
# Utilities
###############################################################################


@contextlib.contextmanager
def inference_context(model):
    """Prepare model for inference"""
    device_type = next(model.parameters()).device.type

    # Prepare model for evaluation
    model.eval()

    # Turn off gradient computation
    with torch.no_grad():

        # Automatic mixed precision on GPU
        with torch.autocast(device_type):
            yield

    # Prepare model for training
    model.train()


def resample(audio, sample_rate, target_rate=emphases.SAMPLE_RATE):
    """Perform audio resampling"""
    if sample_rate == target_rate:
        return audio
    resampler = torchaudio.transforms.Resample(sample_rate, target_rate)
    resampler = resampler.to(audio.device)
    return resampler(audio)
