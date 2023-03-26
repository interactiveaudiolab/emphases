import contextlib
import functools
import json
from pathlib import Path
from typing import List, Optional, Tuple, Type

import pyfoal
import pypar
import torch
import torchaudio
import tqdm

import emphases


###############################################################################
# Emphasis annotation API
###############################################################################


def from_file(
    text_file: List[Path],
    audio_file: List[Path],
    hopsize: float = emphases.HOPSIZE_SECONDS,
    checkpoint: Path = emphases.DEFAULT_CHECKPOINT,
    batch_size: Optional[int] = None,
    pad: bool = False,
    gpu: Optional[int] = None
) -> Tuple[Type[pypar.Alignment], torch.Tensor]:
    """Produce emphasis scores for each word for files on disk

    Args:
        text_file: The speech transcript text file
        audio_file: The speech waveform audio file
        hopsize: The hopsize in seconds
        checkpoint: The model checkpoint to use for inference
        batch_size: The maximum number of frames per batch
        pad: If true, centers frames at hopsize / 2, 3 * hopsize / 2, 5 * ...
        gpu: The index of the gpu to run inference on

    Returns:
        alignment: The forced phoneme alignment
        scores: The float-valued emphasis scores for each word
    """
    # Load text
    with open(text_file, encoding='utf-8') as file:
        text = file.read()

    # Load audio
    audio = emphases.load.audio(audio_file)

    # Detect emphases
    return from_text_and_audio(
        text,
        audio,
        emphases.SAMPLE_RATE,
        hopsize,
        checkpoint,
        batch_size,
        pad,
        gpu)


def from_file_to_file(
    text_file: List[Path],
    audio_file: List[Path],
    output_file: Optional[List[Path]] = None,
    hopsize: float = emphases.HOPSIZE_SECONDS,
    checkpoint: Path = emphases.DEFAULT_CHECKPOINT,
    batch_size: Optional[int] = None,
    pad: bool = False,
    gpu: Optional[int] = None) -> None:
    """Produce emphasis scores for each word for files on disk and save to disk

    Args:
        text_file: The speech transcript text file
        audio_file: The speech waveform audio file
        output_file: The output file. Defaults to text file with json suffix.
        hopsize: The hopsize in seconds
        checkpoint: The model checkpoint to use for inference
        batch_size: The maximum number of frames per batch
        pad: If true, centers frames at hopsize / 2, 3 * hopsize / 2, 5 * ...
        gpu: The index of the gpu to run inference on
    """
    if output_file is None:
        output_file = text_file.with_suffix('.json')

    # Detect emphases
    alignment, results = from_file(
        text_file,
        audio_file,
        hopsize,
        checkpoint,
        batch_size,
        pad,
        gpu)

    # Format results
    results_list = [
        (str(word), word.start(), word.end(), result)
        for word, result in zip(alignment.words(), results)]

    # Save results
    with open(output_file, 'w') as file:
        json.dump(results_list, file, indent=4)


def from_files_to_files(
        text_files: List[Path],
        audio_files: List[Path],
        output_files: Optional[List[Path]] = None,
        hopsize: float = emphases.HOPSIZE_SECONDS,
        checkpoint: Path = emphases.DEFAULT_CHECKPOINT,
        batch_size: Optional[int] = None,
        pad: bool = False,
        gpu: Optional[int] = None) -> None:
    """Produce emphasis scores for each word for many files and save to disk

    Args:
        text_files: The speech transcript text files
        audio_files: The corresponding speech audio files
        output_files: The output files. Default is text files with json suffix.
        hopsize: The hopsize in seconds
        checkpoint: The model checkpoint to use for inference
        batch_size: The maximum number of frames per batch
        pad: If true, centers frames at hopsize / 2, 3 * hopsize / 2, 5 * ...
        gpu: The index of the gpu to run inference on
    """
    # Set default output path
    if output_files is None:
        output_files = [file.with_suffix('.json') for file in text_files]

    # Detect emphases
    annotation_fn = functools.partial(
        from_file_to_file,
        hopsize=hopsize,
        checkpoint=checkpoint,
        batch_size=batch_size,
        pad=pad,
        gpu=gpu)
    for files in iterator(
        zip(text_files, audio_files, output_files),
        emphases.CONFIG,
        len(text_files)
    ):
        annotation_fn(*files)


def from_text_and_audio(
    text: str,
    audio: torch.Tensor,
    sample_rate: int,
    hopsize: float = emphases.HOPSIZE_SECONDS,
    checkpoint: Path = emphases.DEFAULT_CHECKPOINT,
    batch_size: Optional[int] = None,
    pad: bool = False,
    gpu: Optional[int] = None) -> Tuple[Type[pypar.Alignment], torch.Tensor]:
    """Produce emphasis scores for each word

    Args:
        text: The speech transcript
        audio: The speech waveform
        sample_rate: The audio sampling rate
        hopsize: The hopsize in seconds
        checkpoint: The model checkpoint to use for inference
        batch_size: The maximum number of frames per batch
        pad: If true, centers frames at hopsize / 2, 3 * hopsize / 2, 5 * ...
        gpu: The index of the gpu to run inference on

    Returns:
        alignment: The forced phoneme alignment
        scores: The float-valued emphasis scores for each word
    """
    # Get word alignment
    alignment = pyfoal.align(text, audio, sample_rate, 'p2fa')

    # Infer
    scores = from_alignment_and_audio(
        alignment,
        audio,
        sample_rate,
        hopsize,
        checkpoint,
        batch_size,
        pad,
        gpu)

    return alignment, scores


def from_alignment_and_audio(
    alignment: pypar.Alignment,
    audio: torch.Tensor,
    sample_rate: int,
    hopsize: float = emphases.HOPSIZE_SECONDS,
    checkpoint: Path = emphases.DEFAULT_CHECKPOINT,
    batch_size: Optional[int] = None,
    pad: bool = False,
    gpu: Optional[int] = None) -> Tuple[Type[pypar.Alignment], torch.Tensor]:
    """Produce emphasis scores for each word

    Args:
        alignment: The forced phoneme alignment
        audio: The speech waveform
        sample_rate: The audio sampling rate
        hopsize: The hopsize in seconds
        checkpoint: The model checkpoint to use for inference
        batch_size: The maximum number of frames per batch
        pad: If true, centers frames at hopsize / 2, 3 * hopsize / 2, 5 * ...
        gpu: The index of the gpu to run inference on

    Returns:
        scores: The float-valued emphasis scores for each word
    """
    # Neural method
    if emphases.METHOD == 'neural':

        scores = []

        # Preprocess audio
        iterator = preprocess(
            alignment,
            audio,
            sample_rate,
            hopsize,
            batch_size,
            pad,
            gpu)
        for features, word_bounds in iterator:

            # Infer
            scores.append(infer(features, word_bounds, checkpoint).detach()[0])

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


def infer(features, word_bounds, checkpoint=emphases.DEFAULT_CHECKPOINT):
    """Perform model inference to annotate emphases of each word"""
    # Maybe cache model
    if (
        not hasattr(infer, 'model') or
        infer.checkpoint != checkpoint or
        infer.device_type != features.device.type
    ):
        # Maybe initialize model
        model = emphases.Model()

        # Load from disk
        infer.model, *_ = emphases.checkpoint.load(checkpoint, model)
        infer.checkpoint = checkpoint
        infer.device_type = features.device.type

        # Move model to correct device (no-op if devices are the same)
        infer.model = infer.model.to(features.device)

    # Use full sequence lengths
    frame_lengths = torch.ones(
        (1, features.shape[-1]),
        dtype=torch.long,
        device=features.device)
    word_lengths = torch.ones(
        (1, word_bounds.shape[-1]),
        dtype=torch.long,
        device=features.device)

    # Infer
    return infer.model(features, frame_lengths, word_bounds, word_lengths)[0]


def preprocess(
    alignment,
    audio,
    sample_rate,
    hopsize=emphases.HOPSIZE_SECONDS,
    batch_size=None,
    pad=False,
    gpu=None):
    """Convert audio to model input"""
    # Convert hopsize to samples
    hopsize = int(emphases.convert.seconds_to_samples(hopsize))

    # Resample
    if sample_rate != emphases.SAMPLE_RATE:
        audio = resample(audio, sample_rate)

    # Pad audio and get total number of frames
    padding = int((emphases.WINDOW_SIZE - hopsize) / 2)
    if pad:
        audio = torch.nn.functional.pad(audio, (padding, padding))
        total_frames = int(audio.shape[-1] / hopsize)
    else:
        total_frames = int((audio.shape[-1] - 2 * padding) / hopsize)

    # Default to running all frames in a single batch
    batch_size = total_frames if batch_size is None else batch_size

    # Generate batches
    start = 0
    while start < len(alignment):

        # Accumulate enough frames for this batch
        frames = 0
        end = start + 1
        while end < len(alignment):

            # Get duration of this word in frames
            duration = int(emphases.convert.seconds_to_frames(
                alignment[end - 1].duration()))

            # Update frames
            frames += duration

            # Stop if we've accumulated enough frames
            if frames > batch_size:
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

        # Preprocess audio
        batch_features = emphases.data.preprocess.from_audio(
            batch_audio,
            gpu=gpu)

        # Run inference
        yield batch_features, batch_word_bounds

        # Update start word
        start = end


###############################################################################
# Word and frame resolution resampling
###############################################################################


# # TODO - targets should NOT already be resampled to frame rate
# # TODO - move to inference
# # Get center time of each word in frames (we know that the targets are accurate here since they're interpolated from here)
# word_centers = \
#     word_bounds[:, 0] + (word_bounds[:, 1] - word_bounds[:, 0]) // 2

# #Allocate tensors for wordwise scores and targets
# word_scores = torch.zeros(word_centers.shape, device=scores.device)
# word_targets = torch.zeros(word_centers.shape, device=scores.device)
# word_masks = torch.zeros(word_centers.shape, device=scores.device)

# for stem in range(targets.shape[0]): #Iterate over batch
#     stem_word_centers = word_centers[stem]
#     stem_word_targets = targets.squeeze(1)[stem, stem_word_centers]
#     stem_word_mask = torch.where(stem_word_centers == 0, 0, 1)

#     word_targets[stem] = stem_word_targets
#     word_masks[stem] = stem_word_mask

#     for i, (start, end) in enumerate(word_bounds[stem].T):
#         word_outputs = scores.squeeze(1)[stem, start:end]
#         if word_outputs.shape[0] == 0:
#             continue
#         word_score = emphases.frames_to_words(word_outputs)
#         word_scores[stem, i] = word_score

# scores = word_scores
# targets = word_targets
# mask = word_masks


def downsample(x, word_lengths, word_bounds):
    """Interpolate from frame to word resolution"""
    # Average resampling
    if emphases.DOWNSAMPLE_METHOD == 'average':

        # Allocate memory for word resolution sequence
        word_embeddings = torch.zeros(
            (x.shape[0], x.shape[1], word_lengths.max().item()),
            device=x.device)

        # Take average of frames corresponding to each word
        i = 0
        iterator = enumerate(zip(x, word_bounds, word_lengths))
        for i, (embedding, bounds, length) in iterator:
            for j in range(length):
                start, end = bounds[0, j], bounds[1, j]
                word_embeddings[i, :, j] = embedding[:, start:end].mean(dim=1)

    # Maximum resampling
    # TODO - vectorize max and center resampling methods
    if emphases.DOWNSAMPLE_METHOD == 'max':
        if dim is not None:
            max_out = input.max(dim=dim)
            return max_out.values
        return input.max()

    # Centerpoint resampling
    elif emphases.DOWNSAMPLE_METHOD == 'center':

        centers = word
        center_index = torch.Tensor([input.shape[dim] // 2]).int().to(device=input.device)
        return torch.index_select(input, dim, center_index).squeeze()

    else:
        raise ValueError(
            f'Interpolation method {emphases.DOWNSAMPLE_METHOD} is not defined')


def upsample(x, frame_lengths, word_bounds):
    """Interpolate from word to frame resolution"""
    # TODO - batch
    x = x[0]
    frames = frame_lengths[0]
    word_bounds = word_bounds[0]

    # Get center time of each word in frames
    word_centers = (
        word_bounds[0] + (word_bounds[1] - word_bounds[0]) / 2.)[None]

    # Get frame centers
    frame_centers = .5 + torch.arange(frames)[None]

    # Linear interpolation
    if emphases.UPSAMPLE_METHOD == 'linear':

        # Compute slope and intercept at original times
        slope = (
            (x[:, 1:] - x[:, :-1]) /
            (word_times[:, 1:] - word_times[:, :-1]))
        intercept = x[:, :-1] - slope.mul(word_times[:, :-1])

        # Compute indices at which we evaluate points
        indices = torch.sum(
            torch.ge(frame_times[:, :, None], word_times[:, None, :]), -1) - 1
        indices = torch.clamp(indices, 0, slope.shape[-1] - 1)

        # Compute index into parameters
        line_idx = torch.linspace(
            0,
            indices.shape[0],
            1,
            device=indices.device).to(torch.long)
        line_idx = line_idx.expand(indices.shape)

        # Interpolate
        return (
            slope[line_idx, indices].mul(frame_times) +
            intercept[line_idx, indices])

    # Nearest neighbors interpolation
    if emphases.UPSAMPLE_METHOD == 'nearest':

        # Compute indices at which we evaluate points
        indices = torch.sum(
            torch.ge(frame_times[:, :, None], word_times[:, None, :]), -1) - 1
        indices = torch.clamp(indices, 0, word_times.shape[-1] - 1)

        # Get nearest score
        return torch.index_select(x, 1, indices[0])

    raise ValueError(f'Interpolation method {method} is not defined')


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
        if device_type == 'cuda':
            with torch.autocast(device_type):
                yield

        else:
            yield

    # Prepare model for training
    model.train()


def iterator(iterable, message, initial=0, total=None):
    """Create a tqdm iterator"""
    total = len(iterable) if total is None else total
    return tqdm.tqdm(
        iterable,
        desc=message,
        dynamic_ncols=True,
        initial=initial,
        total=total)


def resample(audio, sample_rate, target_rate=emphases.SAMPLE_RATE):
    """Perform audio resampling"""
    if sample_rate == target_rate:
        return audio
    resampler = torchaudio.transforms.Resample(sample_rate, target_rate)
    resampler = resampler.to(audio.device)
    return resampler(audio)
