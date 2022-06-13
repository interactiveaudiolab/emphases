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
emphases of many files. Emphases are saved as a list of words with
start times (`word['start']`), end times (`word['end']`), and a boolean
emphasis flag (`word['emphasized']`).


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

## Test

`pip install pytest && pytest`
