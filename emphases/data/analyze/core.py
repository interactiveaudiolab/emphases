import os
import json
import yaml

import emphases
import torch
import torchaudio
import pypar

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


"""

# TODO: 
# Need more data, the stats definition would take shape as the data grows

Overall stats
- # of words annotated
- # of words marked prominent
- Average duration, std dev of words, audio files
- Average duration, std dev of annotated words
- Average duration/ # words in files with ZERO annotation
- relationship between (# of prominent words in a sentence vs # of words/duration in sentence)

Annotator specific stats
- Average number of words per utterance annotated by each annotator
- Inter-annotator agreement between each pair of annotators, using metrics such as Cohen's kappa or Fleiss' kappa
- Average time taken by each annotator to annotate an utterance

"""

def analysis(datasets):
    for dataset in datasets:
        if dataset == "annotate":
            annotate()
        else:
            raise ValueError(f'Dataset {dataset} is not defined')

class AnnotateStats():
    def __init__(self, stem):
        self.stem = stem
        self.cache_directory = emphases.CACHE_DIR / 'annotate'
        self.alignment = pypar.Alignment(self.cache_directory / 'alignment' / f'{stem}.TextGrid')
        
    def get_duration(self):
        return self.alignment.duration()
    
    def get_word_durations(self):
        bounds = self.alignment.word_bounds(emphases.SAMPLE_RATE)
        return [(bound[1] - bound[0])/emphases.SAMPLE_RATE for bound in bounds]
    
    def get_prom_durations(self, response):
        word_durations = self.get_word_durations()
        return [word_durations[idx] for idx, mark in enumerate(response) if int(mark)]

def annotate():
    stats_data = {}

    # Extract annotations to data directory
    with open(emphases.DEFAULT_ANNOTATION_CONFIG, "r") as stream:
        try:
            annotation_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    data_directory = emphases.DATA_DIR / 'annotate' / annotation_config['name']
    cache_directory = emphases.CACHE_DIR / 'annotate'

    response_file = data_directory / 'tables' / 'responses.csv'
    results_file = data_directory / 'results.json'

    analysis_dir = emphases.EVAL_DIR / 'annotate'
    (analysis_dir).mkdir(exist_ok=True, parents=True)
    
    # Load the responses
    df = pd.read_csv(response_file)

    # Count specific stats
    df['prom_count'] = df['Response'].apply(lambda x: x.count('1'))
    df['word_count'] = df['Response'].apply(lambda x: len(x))

    stats_data['total_words_annotated'] = df.word_count.sum()
    stats_data['total_prominent_words'] = df.prom_count.sum()

    # Duration stats
    df['duration_seconds'] = df['Stem'].apply(lambda x:AnnotateStats(x).get_duration())
    df['word_durations'] = df['Stem'].apply(lambda x:AnnotateStats(x).get_word_durations())
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
    
    # Save analysis statistics
    for key in stats_data:
        if isinstance(stats_data[key], np.float64) or isinstance(stats_data[key], np.int64):
            stats_data[key] = float(stats_data[key])

    with open(analysis_dir / 'analysis.json', 'w') as fp:
        json.dump(stats_data, fp, indent=4)

    print(f"Analysis saved in {analysis_dir}")