# emphases
Representations of prosody for detecting emphases


## Installation

`pip install -e .`

You must also install HTK to use the pyfoal aligner. See
[here](https://github.com/maxrmorrison/pyfoal).


## Usage

### Application programming interface

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

The `alignment` is a `pypar.Alignment` object. You can also use
`emphases.from_text_and_audio` to compute emphases from a string and,
numpy array, `emphases.from_file_to_file` to automatically save
results to disk, or `emphases.from_files_to_files` to compute the
emphases of many files. Emphases are saved as a list of four-tuples
containing the word, start time, end time, and boolean that is true if
the word is emphasized.


### Command-line interface

```
python -m emphases
    text_files [text_files ...]
    audio_files [audio_files ...]
    [--output_file OUTPUT_FILE [OUTPUT_FILE ...]]

Determine which words in a speech file are emphasized

positional arguments:
  text_files            Text file containing transcripts
  audio_files           The corresponding speech audio files

optional arguments:
  -h, --help            show this help message and exit
  --output_file OUTPUT_FILE [OUTPUT_FILE ...]
                        Json files to save results.
                        Defaults to text files with json extension.
```


## Training

### Download data - Done

Complete all TODOs in `data/download/`, then run `python -m emphases.download DATASET`.
`python -m emphases.data.download --datasets Buckeye`

### Partition data

Complete all TODOs in `partition/`, then run `python -m emphases.partition
DATASET`.


### Preprocess data - Done

Complete all TODOs in `preprocess/`, then run `python -m emphases.preprocess
DATASET`. All preprocessed data is saved in `data/cache/DATASET`.

`python -m emphases.preprocess Buckeye`

After processing the Buckeye TextGrid excerpts from Prof. Cole's lab, we are manually removing the tonic and PointTier from textgrid files. And changing the size=2

### Train

Complete all TODOs in `data/` and `model.py`, then run `python -m emphases.train --config <config> --dataset
DATASET --gpus <gpus>`.


### Evaluate

Complete all TODOs in `evaluate/`, then run `python -m emphases.evaluate
--datasets <datasets> --checkpoint <checkpoint> --gpu <gpu>`.


### Monitor

Run `tensorboard --logdir runs/`. If you are running training
remotely, you must create a SSH connection with port forwarding to view
Tensorboard. This can be done with `ssh -L 6006:localhost:6006
<user>@<server-ip-address>`. Then, open `localhost:6006` in your browser.


## Test

`pip install pytest && pytest`


## FAQ

### What is the directory `emphases/assets` for?

This directory is for
[_package data_](https://packaging.python.org/guides/distributing-packages-using-setuptools/#package-data).
When you pip install a package, pip will
automatically copy the python files to the installation folder (in
`site_packages`). Pip will _not_ automatically copy files that are not Python
files. So if your code depends on non-Python files to run (e.g., a pretrained
model, normalizing statistics, or data partitions), you have to manually
specify these files in `setup.py`. This is done for you in this repo. In
general, only small files that are essential at runtime should be placed in
this folder.


### What if my evaluation includes subjective experiments?

In this case, replace the `<file>` argument of `emphases.evaluate` with a
directory. Write any objective metrics to a file within this directory, as well
as any generated files that will be subjectively evaluated.


### How do I release my code so that it can be downloaded via pip?

Code release involves making sure that `setup.py` is up-to-date and then
uploading your code to [`pypi`](https://www.pypi.org).
[Here](https://packaging.python.org/tutorials/packaging-projects/) is a good
tutorial for this process.
