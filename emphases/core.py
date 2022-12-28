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
    for files in tqdm.tqdm(zip(text_files, audio_files, output_files)):
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
    scores = []

    # Get word alignment
    alignment = pyfoal.align(text, audio, sample_rate)

    # Preprocess audio
    iterator = preprocess(
        alignment,
        audio,
        sample_rate,
        hopsize,
        batch_size,
        pad,
        gpu)
    for align, features in iterator:

        # Infer
        scores.append(infer(align, features, checkpoint).detach())

    # Concatenate results
    scores = torch.cat(scores, 1)

    return alignment, scores


###############################################################################
# Utilities
###############################################################################


def infer(alignment, audio, checkpoint=emphases.DEFAULT_CHECKPOINT):
    """Perform model inference to annotate emphases of each word"""
    # Maybe cache model
    if (
        emphases.METHOD in ['framewise', 'wordwise'] and
        (
            not hasattr(infer, 'model') or
            infer.checkpoint != checkpoint or
            infer.device_type != audio.device.type
        )
    ):
        # Maybe initialize model
        model = emphases.Model()

        # Load from disk
        infer.model, *_ = emphases.checkpoint.load(checkpoint, model)
        infer.checkpoint = checkpoint
        infer.device_type = audio.device.type

        # Move model to correct device (no-op if devices are the same)
        infer.model = infer.model.to(audio.device)

    # Infer
    return infer.model(alignment, audio)


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
        end = start
        while end < len(alignment):

            # Get duration of this word in frames
            duration = emphases.convert.seconds_to_frames(
                alignment[end].duration())

            # Stop if we've accumulated enough frames
            if frames + duration > batch_size:
                break

            # Update frames
            frames += duration
            end += 1

        # Slice alignment
        batch_alignment = alignment[start:end]

        # Slice audio
        start_sample = emphases.convert.frames_to_samples(
            alignment[start].start())
        end_sample = emphases.convert.frames_to_samples(
            alignment[end - 1].end())
        batch_audio = audio[:, start_sample, end_sample]

        # Preprocess audio
        batch_features = emphases.preprocess.from_audio(batch_audio, gpu=gpu)

        # Run inference
        yield batch_alignment, batch_features

        # Update start word
        start = end


def resample(audio, sample_rate, target_rate=emphases.SAMPLE_RATE):
    """Perform audio resampling"""
    if sample_rate == target_rate:
        return audio
    resampler = torchaudio.transforms.Resample(sample_rate, target_rate)
    resampler = resampler.to(audio.device)
    return resampler(audio)
