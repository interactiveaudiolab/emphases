<h1 align="center">Crowdsourced and Automatic Speech Prominence Estimation</h1>
<div align="center">

[![PyPI](https://img.shields.io/pypi/v/emphases.svg)](https://pypi.python.org/pypi/emphases)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://static.pepy.tech/badge/emphases)](https://pepy.tech/project/emphases)

Annotation, training, evaluation and inference of speech prominence

[Paper](https://www.maxrmorrison.com/pdfs/morrison2024crowdsourced.pdf) [Website](https://www.maxrmorrison.com/sites/prominence-estimation) [Dataset](https://zenodo.org/records/10402793)

</div>


## Table of contents

- [Installation](#installation)
- [Inference](#inference)
    * [Application programming interface](#application-programming-interface)
        * [`emphases.from_alignment_and_audio`](#emphasesfrom_alignment_and_audio)
        * [`emphases.from_text_and_audio`](#emphasesfrom_text_and_audio)
        * [`emphases.from_file`](#emphasesfrom_file)
        * [`emphases.from_file_to_file`](#emphasesfrom_file_to_file)
        * [`emphases.from_files_to_files`](#emphasesfrom_files_to_files)
    * [Command-line interface](#command-line-interface)
- [Training](#training)
    * [Download](#download)
    * [Annotate](#annotate)
    * [Preprocess](#preprocess)
    * [Partition](#partition)
    * [Train](#train)
    * [Monitor](#monitor)
- [Evaluation](#reproducing-results)
    * [Evaluate](#evaluate)
    * [Analyze](#analyze)
- [Citation](#citation)


## Installation

`pip install emphases`

By default, we use the Penn Phonetic Forced Aligner (P2FA) via the [`pyfoal`](https://github.com/maxrmorrison/pyfoal/)
repo to perform word alignments. This requires installing HTK. See [the HTK
installation instructions](https://github.com/maxrmorrison/pyfoal/tree/main?tab=readme-ov-file#penn-phonetic-forced-aligner-p2fa)
provided by `pyfoal`. Alternatively, you can use a different forced aligner
and either pass the alignment as a [`pypar.Alignment`](https://github.com/maxrmorrison/pypar/tree/main)
object or save the alignment as a `.TextGrid` file.


## Inference

Perform automatic emphasis annotation using our best pretrained model

```python
import emphases

# Text and audio of speech
text_file = 'example.txt'
audio_file = 'example.wav'

# Detect emphases
alignment, prominence = emphases.from_file(text_file, audio_file)

# Check which words were emphasized
for word, score in zip(alignment, prominence[0]):
    print(f'{word} has a prominence of {score}')
```

The `alignment` is a [`pypar.Alignment`](https://github.com/maxrmorrison/pypar)
object.


### Application programming interface

#### `emphases.from_alignment_and_audio`

```python
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
```


#### `emphases.from_text_and_audio`

```python
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
```


#### `emphases.from_file`

```python
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
```


#### `emphases.from_file_to_file`

```python
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
```

Emphases are saved as a list of five-tuples containing the word, start time,
end time, a float-valued emphasis score, and a boolean that is true if the
word is emphasized.


#### `emphases.from_files_to_files`

```python
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
```


### Command-line interface

```
python -m emphases
    [-h]
    --text_files TEXT_FILES [TEXT_FILES ...]
    --audio_files AUDIO_FILES [AUDIO_FILES ...]
    [--output_files OUTPUT_FILES [OUTPUT_FILES ...]]
    [--checkpoint CHECKPOINT]
    [--batch_size BATCH_SIZE]
    [--gpu GPU]

Determine which words in a speech file are emphasized

options:
  -h, --help            show this help message and exit
  --text_files TEXT_FILES [TEXT_FILES ...]
                        The speech transcript text files
  --audio_files AUDIO_FILES [AUDIO_FILES ...]
                        The corresponding speech audio files
  --output_files OUTPUT_FILES [OUTPUT_FILES ...]
                        The output files. Default is text files with json suffix.
  --checkpoint CHECKPOINT
                        The model checkpoint to use for inference
  --batch_size BATCH_SIZE
                        The maximum number of frames per batch
  --gpu GPU             The index of the gpu to run inference on
```


## Training

### Download data

`python -m emphases.download --datasets <datasets>`.

Downloads and uncompresses datasets.

**N.B.** We omit Buckeye for public release. This evaluation dataset can be
made by [downloading Buckeye](https://buckeyecorpus.osu.edu/) and matching
the files to the
[annotations](https://github.com/ProSD-Lab/Prominence-perception-in-English-French-Spanish/).
The process of matching the files to the annotations was done for us and is
tricky to replicate exactly. However, due to licensing restrictions on
Buckeye, we cannot legally distribute our private, aligned annotations.


### Annotate data

Performing annotation requires first installing
[Reproducible Subjective Evaluation (ReSEval)](https://github.com/reseval/reseval).

`python -m emphases.annotate --datasets <datasets>`

Launches a local web application to perform emphasis annotation, according to
the ReSEval configuration file `emphases/assets/configs/annotate.yaml`.
Requires ReSEval to be installed.

`python -m emphases.annotate --datasets <datasets> --remote --production`

Launches a crowdsourced emphasis annotation task, according to the ReSEval
configuration file `emphases/assets/configs/annotate.yaml`. Requires ReSEval
to be installed.


### Partition data

`python -m emphases.partition`

Generates `train`, `valid`, and `test` partitions for all datasets.
Partitioning is deterministic given the same random seed. You do not need to
run this step, as the original partitions are saved in
`emphases/assets/partitions`.


### Preprocess

`python -m emphases.preprocess`


### Train

`python -m emphases.train --config <config> --dataset <dataset> --gpus <gpus>`

Trains a model according to a given configuration. Uses a list of GPU
indices as an argument, and uses distributed data parallelism (DDP)
if more than one index is given. For example, `--gpus 0 3` will train
using DDP on GPUs `0` and `3`.


## Evaluation

### Evaluate

`python -m emphases.evaluate --config <config> --checkpoint <checkpoint> --gpu <gpu>`


### Monitor

Run `tensorboard --logdir runs/`. If you are running training
remotely, you must create a SSH connection with port forwarding to view
Tensorboard. This can be done with `ssh -L 6006:localhost:6006
<user>@<server-ip-address>`. Then, open `localhost:6006` in your browser.


## Citation

### IEEE
M. Morrison, P. Pawar, N. Pruyne, J. Cole, and B. Pardo, "Crowdsourced and Automatic Speech Prominence Estimation," International Conference on Acoustics, Speech, & Signal Processing, 2024.


### BibTex

```
@inproceedings{morrison2024crowdsourced,
    title={Crowdsourced and Automatic Speech Prominence Estimation},
    author={Morrison, Max and Pawar, Pranav and Pruyne, Nathan and Cole, Jennifer and Pardo, Bryan},
    booktitle={International Conference on Acoustics, Speech, & Signal Processing},
    year={2024}
}
