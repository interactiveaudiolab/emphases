"""
Dataset statistics
- # of words annotated
- # of words labeled as emphasized
- Average duration, std dev of words, audio files
- Average duration, std dev of annotated words
- Average duration/ # words in files with ZERO annotation
- relationship between (# of emphasized words in a sentence vs # of words/duration in sentence)

Annotator statistics
- Average number of words per utterance annotated by each annotator
- Inter-annotator agreement between each pair of annotators, using metrics such
  as Cohen's kappa or Fleiss' kappa
- Average time taken by each annotator to annotate an utterance
"""
import json
import yaml

import emphases
import pypar

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


###############################################################################
# Analyze emphasis annotation data
###############################################################################


def analysis(datasets):
    for dataset in datasets:
        if dataset == 'crowdsource':
            crowdsource()
        else:
            raise ValueError(f'Dataset {dataset} is not defined')


###############################################################################
# Utilities
###############################################################################


class AnnotateStats():
    def __init__(self, stem):
        self.stem = stem
        self.cache_directory = emphases.CACHE_DIR / 'crowdsource'
        self.alignment = pypar.Alignment(self.cache_directory / 'alignment' / f'{stem}.TextGrid')

    def duration(self):
        return self.alignment.duration()

    def word_durations(self):
        return [word.duration() for word in self.alignment]

    def get_prom_durations(self, response):
        word_durations = self.word_durations()
        return [
            word_durations[idx] for idx, mark in enumerate(response)
            if int(mark)]

def crowdsource():
    stats_data = {}

    # Extract annotations to data directory
    with open(emphases.DEFAULT_ANNOTATION_CONFIG, "r") as stream:
        try:
            annotation_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    data_directory = emphases.DATA_DIR / 'crowdsource' / annotation_config['name']

    response_file = data_directory / 'tables' / 'responses.csv'

    analysis_dir = emphases.EVAL_DIR / 'crowdsource'
    analysis_dir.mkdir(exist_ok=True, parents=True)

    # Load the responses
    df = pd.read_csv(response_file)

    # Count specific stats
    df['prom_count'] = df['Response'].apply(lambda x: x.count('1'))
    df['word_count'] = df['Response'].apply(lambda x: len(x))

    stats_data['total_words_annotated'] = df.word_count.sum()
    stats_data['total_prominent_words'] = df.prom_count.sum()

    # Duration stats
    df['duration_seconds'] = df['Stem'].apply(lambda x:AnnotateStats(x).duration())
    df['word_durations'] = df['Stem'].apply(lambda x:AnnotateStats(x).word_durations())
    df['prom_durations'] = df.apply(lambda x:AnnotateStats(x['Stem']).get_prom_durations(x['Response']), axis=1)

    # get word and prom word duration stats
    all_word_durations = []
    _ = [all_word_durations.extend(item) for item in df['word_durations'].to_list()]
    stats_data['word_duration_mean'] = np.mean(all_word_durations)
    stats_data['word_duration_std'] = np.std(all_word_durations)

    all_prom_durations = []
    _ = [all_prom_durations.extend(item) for item in df['prom_durations'].to_list()]
    stats_data['prom_duration_mean'] = np.mean(all_prom_durations)
    stats_data['prom_duration_std'] = np.std(all_prom_durations)

    # Prominent word stats
    stats_data['utterance_length_mean'] = df['Response'].str.len().mean()
    stats_data['utterance_length_std'] = df['Response'].str.len().std()
    stats_data['avg_prominent_words_per_utterance'] = df['Response'].apply(lambda x: sum(int(d) for d in str(x))).mean()

    # Utterance length vs # of marked prominent words
    fig = plt.figure(figsize=(4,4))
    x = df['word_count']
    y = df['prom_count']

    # Save plots
    plt.scatter(x, y)
    plt.xlabel('Utterance length')
    plt.ylabel('Number of prominent words per utterance')
    fig.savefig(analysis_dir / 'prom_vs_utter_len.png', dpi=fig.dpi)

    # Get stats to save
    for key in stats_data:
        if (
            isinstance(stats_data[key], np.float64) or
            isinstance(stats_data[key], np.int64)
        ):
            stats_data[key] = float(stats_data[key])

    # Save stats
    with open(analysis_dir / 'analysis.json', 'w') as fp:
        json.dump(stats_data, fp, indent=4)
