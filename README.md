<h1 align="center">emphases</h1>
<div align="center">

[![PyPI](https://img.shields.io/pypi/v/emphases.svg)](https://pypi.python.org/pypi/emphases)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://pepy.tech/badge/emphases)](https://pepy.tech/project/emphases)

</div>

Official code for the paper _Datasets and Scaling Laws for Neural Emphasis Prediction_


## Table of contents

- [Installation](#installation)
- [Inference](#inference)
    * [Application programming interface](#application-programming-interface)
        * [`emphases.from_audio`](#emphasesfrom_audio)
        * [`emphases.from_file`](#emphasesfrom_file)
        * [`emphases.from_file_to_file`](#emphasesfrom_file_to_file)
        * [`emphases.from_files_to_files`](#emphasesfrom_files_to_files)
    * [Command-line interface](#command-line-interface)
- [Training](#training)
    * [Download](#download)
    * [Preprocess](#preprocess)
    * [Partition](#partition)
    * [Train](#train)
    * [Monitor](#monitor)
- [Evaluation](#reproducing-results)
    * [Evaluate](#evaluate)
    * [Analyze](#analyze)
    * [Plot](#plot)
- [Citation](#citation)


## Installation

`pip install -e .`

You must also install HTK to use the pyfoal aligner. See
[here](https://github.com/maxrmorrison/pyfoal).


## Inference

Perform automatic emphasis annotation using our best pretrained model

```
import emphases

# Text and audio of speech
text_file = 'example.txt'
audio_file = 'example.wav'

# Detect emphases
alignment, results = emphases.from_file(text_file, audio_file)

# Check which words were emphasized
for word, result in zip(alignment.words(), results):
    if result:
        print(f'{word} was emphasized')
```

The `alignment` is a [`pypar.Alignment`](https://github.com/maxrmorrison/pypar)
object.


### Application programming interface

#### `emphases.from_text_and_audio`

```
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
```


#### `emphases.from_file`

```
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
```


#### `emphases.from_file_to_file`

```
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
```

Emphases are saved as a list of five-tuples containing the word, start time,
end time, a float-valued emphasis score, and a boolean that is true if the
word is emphasized.


#### `emphases.from_files_to_files`

```
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
```


### Command-line interface

```
python -m emphases
    [-h]
    --text_files TEXT_FILES [TEXT_FILES ...]
    --audio_files AUDIO_FILES [AUDIO_FILES ...]
    [--output_files OUTPUT_FILES [OUTPUT_FILES ...]]
    [--hopsize HOPSIZE]
    [--checkpoint CHECKPOINT]
    [--batch_size BATCH_SIZE]
    [--pad]
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
  --hopsize HOPSIZE     The hopsize in seconds
  --checkpoint CHECKPOINT
                        The model checkpoint to use for inference
  --batch_size BATCH_SIZE
                        The maximum number of frames per batch
  --pad                 If true, centers frames at hopsize / 2, 3 * hopsize / 2, 5 * ...
  --gpu GPU             The index of the gpu to run inference on
```


## Training

### Download data

`python -m emphases.download`.

Downloads and uncompresses datasets.


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


### Plot

**TODO**


### Monitor

Run `tensorboard --logdir runs/`. If you are running training
remotely, you must create a SSH connection with port forwarding to view
Tensorboard. This can be done with `ssh -L 6006:localhost:6006
<user>@<server-ip-address>`. Then, open `localhost:6006` in your browser.


## Test

`pip install pytest && pytest`


## Citation

### IEEE
P. Pawar, M. Morrison, N. Pruyne, J. Cole, and B. Pardo, "Datasets and Scaling Laws for Neural Emphasis Prediction," Interspeech, <TODO - month> 2023.


### BibTex

```
@inproceedings{pawar2023datasets,
    title={Datasets and Scaling Laws for Neural Emphasis Prediction},
    author={Pawar, Pranav and Morrison, Max and Pruyne, Nathan and Cole, Jennifer and Pardo, Bryan},
    booktitle={Interspeech},
    month={TODO},
    year={2023}
}
